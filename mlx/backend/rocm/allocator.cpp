// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/memory.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cassert>
#include <sstream>

namespace mlx::core {

namespace rocm {

constexpr int page_size = 16384;

// Any allocations smaller than this will try to use the small pool
constexpr int small_block_size = 8;

// The small pool size in bytes. This should be a multiple of the host page
// size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

// Check if ROCm device is available
static bool rocm_available() {
  static int available = -1;
  if (available < 0) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    available = (err == hipSuccess && device_count > 0) ? 1 : 0;
  }
  return available == 1;
}

// Check if managed memory (HMM) is supported on this device.
// On integrated GPUs (Strix Halo), HMM is actually fast since there's no
// discrete VRAM — managed memory avoids the overhead of hipExtMallocWithFlags.
static bool managed_memory_supported() {
  static int supported = -1;
  if (supported < 0) {
    if (!rocm_available()) {
      supported = 0;
    } else {
      void* test_ptr = nullptr;
      hipError_t err = hipMallocManaged(&test_ptr, 64);
      if (err == hipSuccess) {
        (void)hipFree(test_ptr);
        supported = 1;
      } else {
        supported = 0;
      }
    }
  }
  return supported == 1;
}

static bool is_integrated() {
  static int integrated = -1;
  if (integrated < 0) {
    if (!rocm_available()) {
      integrated = 0;
    } else {
      int device = 0;
      (void)hipGetDevice(&device);
      hipDeviceProp_t props;
      hipError_t err = hipGetDeviceProperties(&props, device);
      integrated = (err == hipSuccess && props.integrated == 1) ? 1 : 0;
    }
  }
  return integrated == 1;
}

inline void* rocm_unified_malloc(size_t size, bool& is_managed) {
  void* data = nullptr;
  hipError_t err;
  if (is_integrated()) {
    // Unified memory device (iGPU/APU): CPU and GPU share system RAM.
    // Try hipExtMallocWithFlags first (fine-grained coherent, best GPU
    // bandwidth). Falls back to hipMallocManaged for large allocations
    // that exceed the small device-local VRAM (~2GB).
    err = hipExtMallocWithFlags(&data, size, hipDeviceMallocFinegrained);
    if (err != hipSuccess) {
      err = hipMallocManaged(&data, size);
    }
    is_managed = true;
  } else if (managed_memory_supported()) {
    err = hipMallocManaged(&data, size);
    is_managed = true;
  } else {
    err = hipHostMalloc(&data, size, hipHostMallocDefault);
    is_managed = false;
  }
  if (err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipMalloc (unified) failed: " << hipGetErrorString(err) << ".";
    throw std::runtime_error(oss.str());
  }
  return data;
}

inline void rocm_unified_free(void* data, bool is_managed) {
  if (is_managed) {
    (void)hipFree(data);
  } else {
    (void)hipHostFree(data);
  }
}

SmallSizePool::SmallSizePool()
    : buffer_(nullptr), data_(nullptr), next_free_(nullptr) {
  if (!rocm_available()) {
    return;
  }

  auto num_blocks = small_pool_size / small_block_size;
  buffer_ = new Block[num_blocks];

  next_free_ = buffer_;

  try {
    data_ = rocm_unified_malloc(small_pool_size, is_managed_);
  } catch (...) {
    delete[] buffer_;
    buffer_ = nullptr;
    next_free_ = nullptr;
    data_ = nullptr;
    return;
  }

  auto curr = next_free_;
  for (size_t i = 1; i < num_blocks; ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  if (data_) {
    rocm_unified_free(data_, is_managed_);
  }
  if (buffer_) {
    delete[] buffer_;
  }
}

RocmBuffer* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  uint64_t i = next_free_ - buffer_;
  next_free_ = next_free_->next;
  b->buf.data = static_cast<char*>(data_) + i * small_block_size;
  b->buf.size = small_block_size;
  b->buf.is_managed = is_managed_;
  b->buf.device = -1;
  return &b->buf;
}

void SmallSizePool::free(RocmBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(RocmBuffer* buf) {
  if (!buffer_) {
    return false;
  }
  constexpr int num_blocks = (small_pool_size / small_block_size);
  auto b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  return block_num >= 0 && block_num < num_blocks;
}

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          page_size,
          [](RocmBuffer* buf) { return buf->size; },
          [this](RocmBuffer* buf) { rocm_free(buf); }),
      memory_limit_(0),
      max_pool_size_(0),
      active_memory_(0),
      peak_memory_(0) {
  if (!rocm_available()) {
    return;
  }

  size_t free, total;
  hipError_t err = hipMemGetInfo(&free, &total);
  if (err == hipSuccess) {
    memory_limit_ = total * 0.8;
    max_pool_size_ = memory_limit_;
  }
}

Buffer RocmAllocator::malloc(size_t size) {
  if (!rocm_available()) {
    throw std::runtime_error(
        "Cannot allocate ROCm memory: no ROCm-capable device detected. "
        "Please use CPU backend instead.");
  }

  // Find available buffer from cache.
  // Use aggressive size rounding to maximize cache hit rate:
  // - Small (<=8B): scalar pool
  // - Medium (<16KB): power-of-2
  // - Large (<1MB): 16KB page aligned
  // - Very large (>=1MB): power-of-2 (coarser buckets = more cache hits)
  // The power-of-2 rounding for large allocations is critical for decode —
  // without it, slightly different sizes (e.g., 1.01MB vs 1.02MB) miss the
  // cache and trigger hipExtMallocWithFlags at ~7ms each.
  auto orig_size = size;
  std::unique_lock lock(mutex_);
  if (size <= small_block_size) {
    size = 8;
  } else if (size < page_size) {
    size = next_power_of_2(size);
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // If we have a lot of memory pressure try to reclaim memory from the cache.
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }

    // Try the scalar pool first
    if (size <= small_block_size) {
      buf = scalar_pool_.malloc();
    }
    lock.unlock();
    if (!buf) {
      if (is_integrated()) {
        // Integrated GPU: allocate unified memory (CPU+GPU accessible).
        // device=-1 signals unified memory — no move_to_unified_memory needed.
        bool is_managed = false;
        void* data = rocm_unified_malloc(size, is_managed);
        buf = new RocmBuffer{data, size, is_managed, -1};
      } else {
        int device = 0;
        hipGetDevice(&device);
        buf = new RocmBuffer{nullptr, size, false, device};
        hipError_t err = hipMalloc(&buf->data, size);

        if (err != hipSuccess) {
          delete buf;
          std::ostringstream oss;
          oss << "hipMalloc failed: " << hipGetErrorString(err) << ".";
          throw std::runtime_error(oss.str());
        }
      }
    }
    lock.lock();
  }
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain the cache below the requested limit.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return Buffer{buf};
}

void RocmAllocator::free(Buffer buffer) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    rocm_free(buf);
  }
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

// This must be called with mutex_ acquired
void RocmAllocator::rocm_free(RocmBuffer* buf) {
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    if (buf->device == -1) {
      rocm_unified_free(buf->data, buf->is_managed);
    } else {
      (void)hipFree(buf->data);
    }
    delete buf;
  }
}

void RocmAllocator::move_to_unified_memory(RocmBuffer& buf) {
  if (buf.device == -1) {
    return;
  }
  bool is_managed = false;
  void* data = rocm_unified_malloc(buf.size, is_managed);
  
  // Use default memcpy to sync from VRAM to Host/Managed
  hipError_t err = hipMemcpy(data, buf.data, buf.size, hipMemcpyDefault);
  if (err != hipSuccess) {
    rocm_unified_free(data, is_managed);
    std::ostringstream oss;
    oss << "hipMemcpy failed: " << hipGetErrorString(err) << ".";
    throw std::runtime_error(oss.str());
  }
  
  // Free the VRAM buffer
  (void)hipFree(buf.data);
  
  // Update the buffer to point to the new unified memory
  buf.data = data;
  buf.is_managed = is_managed;
  buf.device = -1;
}

size_t RocmAllocator::get_active_memory() const {
  return active_memory_;
}

size_t RocmAllocator::get_peak_memory() const {
  return peak_memory_;
}

void RocmAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t RocmAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t RocmAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, memory_limit_);
  return limit;
}

size_t RocmAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void RocmAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
}

RocmAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of RocmAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
}

} // namespace rocm

namespace allocator {

Allocator& allocator() {
  return rocm::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto& cbuf = *static_cast<rocm::RocmBuffer*>(ptr_);

  if (cbuf.device == -1) {
    // Unified memory (integrated GPU or hipMallocManaged): CPU-accessible.
    // hipStreamSynchronize(nullptr) waits for the default stream — lighter
    // than hipDeviceSynchronize which waits for ALL streams.
    (void)hipStreamSynchronize(nullptr);
  } else {
    // Discrete GPU VRAM: full sync + migrate to host-accessible memory.
    (void)hipDeviceSynchronize();
    rocm::allocator().move_to_unified_memory(cbuf);
  }
  return cbuf.data;
}

} // namespace allocator

size_t get_active_memory() {
  return rocm::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return rocm::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return rocm::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return rocm::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return rocm::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return rocm::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return rocm::allocator().set_cache_limit(limit);
}
void clear_cache() {
  rocm::allocator().clear_cache();
}

// Not supported in ROCm.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core

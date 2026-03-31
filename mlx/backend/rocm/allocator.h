// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace mlx::core::rocm {

using allocator::Buffer;

struct RocmBuffer {
  void* data;
  size_t size;
  bool is_managed;
  int device;
};

// ---------------------------------------------------------------------------
// SizeClassPool — fixed-size block pool with free list
// ---------------------------------------------------------------------------

class SizeClassPool {
 public:
  SizeClassPool() = default;
  ~SizeClassPool();

  SizeClassPool(const SizeClassPool&) = delete;
  SizeClassPool& operator=(const SizeClassPool&) = delete;

  void init(size_t block_size, size_t slab_page_size);
  RocmBuffer* malloc();
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf) const;
  bool grow();

  size_t block_size() const { return block_size_; }
  size_t free_count() const { return free_count_; }
  size_t total_allocated() const { return backing_pages_.size() * slab_page_size_; }
  size_t free_memory() const { return free_count_ * block_size_; }
  bool initialized() const { return block_size_ > 0; }

 private:
  union Block {
    Block* next;
    RocmBuffer buf;
  };

  size_t block_size_{0};
  size_t slab_page_size_{0};
  bool is_managed_{false};

  std::vector<void*> backing_pages_;
  std::vector<Block*> block_arrays_;
  std::vector<size_t> blocks_per_page_;

  Block* next_free_{nullptr};
  size_t free_count_{0};
  size_t total_blocks_{0};
};

// ---------------------------------------------------------------------------
// SlabAllocator — multi-tier slab allocator for sizes <= 1MB
// ---------------------------------------------------------------------------

class SlabAllocator {
 public:
  static constexpr int kNumSizeClasses = 18;
  static constexpr size_t kMaxSlabSize = 1 << 20;

  SlabAllocator();
  ~SlabAllocator() = default;

  RocmBuffer* malloc(size_t size);
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf) const;
  bool grow(size_t size);
  void warmup();

  size_t total_allocated() const;
  size_t free_memory() const;

  static int size_class_index(size_t size);
  static size_t round_to_size_class(size_t size);

 private:
  SizeClassPool pools_[kNumSizeClasses];
};

// ---------------------------------------------------------------------------
// RocmAllocator
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  void move_to_unified_memory(RocmBuffer& buf);

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  void rocm_free(RocmBuffer* buf);

  RocmAllocator();
  friend RocmAllocator& allocator();

  std::mutex mutex_;
  size_t memory_limit_;
  size_t max_pool_size_;
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SlabAllocator slab_allocator_;
};

RocmAllocator& allocator();

} // namespace mlx::core::rocm

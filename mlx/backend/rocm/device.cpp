// Copyright © 2025 Apple Inc.

#include "mlx/backend/hip/device.h"
#include "mlx/backend/hip/worker.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>
#include <future>
#include <unordered_set>

namespace mlx::core::cu {

namespace {

bool use_hip_graphs() {
  static bool use_graphs = env::get_var("MLX_USE_HIP_GRAPHS", true);
  return use_graphs;
}

const char* save_hip_graphs_dot_file() {
  static const char* filename = []() -> const char* {
    const char* env = std::getenv("MLX_SAVE_HIP_GRAPHS_DOT_FILE");
    if (env && std::strlen(env) == 0) {
      return nullptr;
    }
    return env;
  }();
  return filename;
}

inline bool is_empty_dim(dim3 dim) {
  return (dim.x == 0 && dim.y == 0 && dim.z == 0) ||
      (dim.x == 1 && dim.y == 1 && dim.z == 1);
}

} // namespace

Device::Device(int device) : device_(device) {
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &compute_capability_major_, hipDeviceAttributeComputeCapabilityMajor, device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &compute_capability_minor_, hipDeviceAttributeComputeCapabilityMinor, device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &concurrent_managed_access_,
      hipDeviceAttributeConcurrentManagedAccess,
      device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &host_native_atomic_, hipDeviceAttributeHostNativeAtomicSupported, device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &managed_memory_, hipDeviceAttributeManagedMemory, device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &memory_pools_, hipDeviceAttributeMemoryPoolsSupported, device_));
}

Device::~Device() = default;

void Device::make_current() {
  // We need to set/get current HIP device very frequently, cache it to reduce
  // actual calls of HIP APIs. Use -1 as sentinel so the first call on each
  // new thread always calls hipSetDevice (which establishes the HIP primary
  // context). Without this, device 0 would never get set on a new thread.
  static thread_local int current = -1;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

CommandEncoder::CaptureContext::CaptureContext(CommandEncoder& enc) : enc(enc) {
  enc.device().make_current();
  if (!use_hip_graphs()) {
    return;
  }
  CHECK_HIP_ERROR(
      hipStreamBeginCapture(enc.stream(), hipStreamCaptureModeThreadLocal));
}

CommandEncoder::CaptureContext::~CaptureContext() {
  if (!use_hip_graphs()) {
    enc.node_count_++;
    return;
  }

  graph.end_capture(enc.stream());
  if (discard) {
    return;
  }
  enc.add_graph_node(graph);
}

CommandEncoder::ConcurrentContext::ConcurrentContext(CommandEncoder& enc)
    : enc(enc) {
  enc.in_concurrent_ = true;
}

CommandEncoder::ConcurrentContext::~ConcurrentContext() {
  enc.in_concurrent_ = false;
  if (!use_hip_graphs()) {
    return;
  }

  // Use an empty graph node for synchronization
  CommandEncoder::GraphNode empty{NULL, "E", std::to_string(enc.node_count_++)};
  CHECK_HIP_ERROR(hipGraphAddEmptyNode(&empty.node, enc.graph_, NULL, 0));

  // Insert the concurrent -> empty node dependencies
  for (auto& from : enc.concurrent_nodes_) {
    enc.from_nodes_.push_back(from.node);
    enc.to_nodes_.push_back(empty.node);
    enc.graph_deps_key_ += from.id;
    enc.graph_deps_key_ += "-";
    enc.graph_deps_key_ += empty.id;
    enc.graph_deps_key_ += "-";
  }

  // Insert the input -> concurrent node dependencies without updating output
  // nodes
  auto outputs = std::move(enc.active_outputs_);
  enc.insert_graph_dependencies(std::move(enc.concurrent_nodes_));

  // Update output node to be the empty node
  for (auto o : outputs) {
    enc.node_map_.emplace(o, empty).first->second = empty;
  }
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  node.id = std::to_string(node_count_++);
  if (in_concurrent_) {
    concurrent_nodes_.push_back(std::move(node));
  } else {
    std::vector<GraphNode> nodes;
    nodes.push_back(std::move(node));
    insert_graph_dependencies(std::move(nodes));
  }
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  {
    // Dependencies must be added in the same order to produce a consistent
    // topology
    std::unordered_set<hipGraphNode_t> set_deps;
    for (auto d : active_deps_) {
      if (auto it = node_map_.find(d); it != node_map_.end()) {
        auto [_, inserted] = set_deps.insert(it->second.node);
        if (inserted) {
          deps.push_back(it->second);
        }
      }
    }
  }
  active_deps_.clear();

  for (auto o : active_outputs_) {
    for (auto& node : nodes) {
      node_map_.emplace(o, node).first->second = node;
    }
  }
  active_outputs_.clear();

  for (auto& from : deps) {
    for (auto& to : nodes) {
      from_nodes_.push_back(from.node);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += from.id;
      graph_deps_key_ += "-";
      graph_deps_key_ += to.id;
      graph_deps_key_ += "-";
    }
  }
}

// Can be tuned with MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER
std::pair<int, int> get_graph_limits(Device& d) {
  auto cc =
      d.compute_capability_major() * 100 + d.compute_capability_minor() * 10;
  int ops = 20;
  int mb = 100;
  switch (cc) {
    case 800: // A100
      ops = 20;
      mb = 400;
      break;
    case 900: // H100
    case 1000: // B200
    case 1200: // Consumer Blackwell
      ops = 100;
      mb = 1000;
      break;
    case 1210: // DGX Spark
      ops = 20;
      mb = 25;
      break;
  }
  return {env::max_ops_per_buffer(ops), env::max_mb_per_buffer(mb)};
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      graph_(d),
      worker_(std::make_shared<Worker>(d)),
      graph_cache_("MLX_HIP_GRAPH_CACHE_SIZE", /* default_capacity */ 400) {
  std::tie(max_ops_per_graph_, max_mb_per_graph_) = get_graph_limits(d);
  worker_->start();
}

CommandEncoder::~CommandEncoder() {
  synchronize();
  worker_->stop();
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  bytes_in_graph_ += arr.data_size();
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }

  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::add_kernel_node_raw(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    dim3 cluster_dim,
    uint32_t smem_bytes,
    void** params) {
  bool use_cluster = !is_empty_dim(cluster_dim);
  assert(!use_cluster || device_.compute_capability_major() >= 9);

  if (!use_hip_graphs()) {
    node_count_++;
    hipLaunchConfig_t config = {};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream();
    hipLaunchAttribute attr = {};
    if (use_cluster) {
      attr.id = hipLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = cluster_dim.x;
      attr.val.clusterDim.y = cluster_dim.y;
      attr.val.clusterDim.z = cluster_dim.z;
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    CHECK_HIP_ERROR(hipLaunchKernelExC(&config, func, params));
    return;
  }

  hipKernelNodeParams kernel_params = {0};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  hipGraphNode_t node = add_kernel_node_raw(kernel_params);
  if (use_cluster) {
    hipKernelNodeAttrValue attr = {};
    attr.clusterDim.x = cluster_dim.x;
    attr.clusterDim.y = cluster_dim.y;
    attr.clusterDim.z = cluster_dim.z;
    CHECK_HIP_ERROR(hipGraphKernelNodeSetAttribute(
        node, hipLaunchAttributeClusterDimension, &attr));
  }
}

void CommandEncoder::add_kernel_node_raw(
    hipFunction_t func,
    dim3 grid_dim,
    dim3 block_dim,
    dim3 cluster_dim,
    uint32_t smem_bytes,
    void** params) {
  bool use_cluster = !is_empty_dim(cluster_dim);
  assert(!use_cluster || device_.compute_capability_major() >= 9);

  if (!use_hip_graphs()) {
    node_count_++;
    HIP_LAUNCH_CONFIG config = {};
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_bytes;
    config.hStream = stream();
    hipLaunchAttribute attr = {};
    if (use_cluster) {
      attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attr.value.clusterDim.x = cluster_dim.x;
      attr.value.clusterDim.y = cluster_dim.y;
      attr.value.clusterDim.z = cluster_dim.z;
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    CHECK_HIP_ERROR(hipDrvLaunchKernelEx(&config, func, params, nullptr));
    return;
  }

  hipKernelNodeParams kernel_params = {};
  kernel_params.func = func;
  kernel_params.gridDimX = grid_dim.x;
  kernel_params.gridDimY = grid_dim.y;
  kernel_params.gridDimZ = grid_dim.z;
  kernel_params.blockDimX = block_dim.x;
  kernel_params.blockDimY = block_dim.y;
  kernel_params.blockDimZ = block_dim.z;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  hipGraphNode_t node = add_kernel_node_raw(kernel_params);
  if (use_cluster) {
    hipLaunchAttributeValue attr = {};
    attr.clusterDim.x = cluster_dim.x;
    attr.clusterDim.y = cluster_dim.y;
    attr.clusterDim.z = cluster_dim.z;
    CHECK_HIP_ERROR(hipGraphKernelNodeSetAttribute(
        node, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &attr));
  }
}

hipGraphNode_t CommandEncoder::add_kernel_node_raw(
    const hipKernelNodeParams& params) {
  hipGraphNode_t node;
  CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, graph_, NULL, 0, &params));
  insert_graph_dependencies(GraphNode{node, "K"});
  return node;
}

hipGraphNode_t CommandEncoder::add_kernel_node_raw(
    const hipKernelNodeParams& params) {
  hipGraphNode_t node;
  CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, graph_, NULL, 0, &params));
  insert_graph_dependencies(GraphNode{node, "K"});
  return node;
}

std::pair<std::string, bool> subgraph_to_key(hipGraph_t graph) {
  // Constructs a key representing the nodes of a sub-graph.
  // Also checks if the sub-graph is updatable as HIP graphs do not get
  // updated correctly if a kernel node getting updated has a different cluster
  // shape than the node it's being updated with.
  std::string key = "(";
  size_t num_nodes = 0;
  CHECK_HIP_ERROR(hipGraphGetNodes(graph, nullptr, &num_nodes));
  if (num_nodes == 0) {
    return {key + ")", true};
  }
  bool is_updatable = true;
  std::vector<hipGraphNode_t> nodes(num_nodes);
  CHECK_HIP_ERROR(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
  for (const auto& node : nodes) {
    if (!is_updatable) {
      break;
    }
    hipGraphNodeType type;
    CHECK_HIP_ERROR(hipGraphNodeGetType(node, &type));
    switch (type) {
      case hipGraphNodeTypeGraph: {
        // Try to be updatable for a structure like graph -> graph -> kernel
        hipGraph_t child;
        CHECK_HIP_ERROR(hipGraphChildGraphNodeGetGraph(node, &child));
        auto [subkey, sub_is_updatable] = subgraph_to_key(child);
        is_updatable &= sub_is_updatable;
        key += subkey;
        break;
      }
      case hipGraphNodeTypeHost:
        key += "H";
        break;
      case hipGraphNodeTypeMemset:
        key += "M";
        break;
      case hipGraphNodeTypeKernel: {
        hipLaunchAttributeValue cluster_dim;
        CHECK_HIP_ERROR(hipGraphKernelNodeGetAttribute(
            node, hipLaunchAttributeClusterDimension, &cluster_dim));
        // Only allow dim.x to be greater than 1
        if (cluster_dim.clusterDim.y > 1 || cluster_dim.clusterDim.z > 1) {
          is_updatable = false;
        } else {
          key += "K";
          key += std::to_string(cluster_dim.clusterDim.x);
        }
        break;
      }
      case hipGraphNodeTypeWaitEvent:
        key += "W";
        break;
      case hipGraphNodeTypeEventRecord:
        key += "R";
        break;
      default:
        is_updatable = false;
    }
  }
  key += ")";
  return {key, is_updatable};
}

void CommandEncoder::add_graph_node(hipGraph_t child) {
  if (!use_hip_graphs()) {
    node_count_++;
    CudaGraphExec graph_exec;
    graph_exec.instantiate(child);
    device_.make_current();
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream()));
    return;
  }
  hipGraphNode_t node;
  auto [sub_graph_key, is_updatable] = subgraph_to_key(child);
  is_graph_updatable_ &= is_updatable;
  CHECK_HIP_ERROR(hipGraphAddChildGraphNode(&node, graph_, NULL, 0, child));
  insert_graph_dependencies(GraphNode{node, sub_graph_key});
}

void CommandEncoder::add_graph_node(
    hipGraph_t child,
    const std::string& subgraph_key,
    bool is_updatable) {
  if (!use_hip_graphs()) {
    node_count_++;
    CudaGraphExec graph_exec;
    graph_exec.instantiate(child);
    device_.make_current();
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream()));
    return;
  }
  is_graph_updatable_ &= is_updatable;
  hipGraphNode_t node;
  CHECK_HIP_ERROR(hipGraphAddChildGraphNode(&node, graph_, NULL, 0, child));
  insert_graph_dependencies(GraphNode{node, subgraph_key});
}

bool CommandEncoder::needs_commit() {
  return (node_count_ > max_ops_per_graph_) ||
      ((bytes_in_graph_ >> 20) > max_mb_per_graph_);
}

void CommandEncoder::commit() {
  nvtx3::scoped_range r("CommandEncoder::commit");
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  if (use_hip_graphs() && node_count_ > 0) {
    if (!from_nodes_.empty()) {
#if HIPRT_VERSION >= 13000
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          nullptr, // edgeData
          from_nodes_.size()));
#else
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          graph_, from_nodes_.data(), to_nodes_.data(), from_nodes_.size()));
#endif
    }

    device_.make_current();

    if (!is_graph_updatable_) {
      CudaGraphExec graph_exec;
      graph_exec.instantiate(graph_);
      CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    } else {
      auto graph_key = graph_nodes_key_ + ":" + graph_deps_key_;
      auto& graph_exec = graph_cache_[graph_key];

      if (graph_exec != nullptr) {
        hipGraphExecUpdateResult update_result;
#if HIPRT_VERSION >= 12000
        hipGraphExecUpdateResultInfo info;
        hipGraphExecUpdate(graph_exec, graph_, &info);
        update_result = info.result;
#else
        hipGraphNode_t error_node;
        hipGraphExecUpdate(graph_exec, graph_, &error_node, &update_result);
#endif // HIPRT_VERSION >= 12000
        if (update_result != hipGraphExecUpdateSuccess) {
          hipGetLastError(); // reset error
          graph_exec.reset();
        }
      }
      if (graph_exec == nullptr) {
        graph_exec.instantiate(graph_);
      }

      CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    }

    // Save hip graph to dot file
    if (const char* filename = save_hip_graphs_dot_file(); filename) {
      static int count = 0;
      auto path = fmt::format("{}_{}.dot", filename, ++count);
      CHECK_HIP_ERROR(hipGraphDebugDotPrint(graph_, path.c_str(), 0));
    }

    // Reset state
    from_nodes_.clear();
    to_nodes_.clear();
    graph_deps_key_.clear();
    graph_nodes_key_.clear();
    node_map_.clear();
    graph_ = CudaGraph(device_);
    is_graph_updatable_ = true;
  }

  // Put completion handlers in a batch.
  worker_->commit(stream_);
  node_count_ = 0;
  bytes_in_graph_ = 0;
}

void CommandEncoder::synchronize() {
  CHECK_HIP_ERROR(hipStreamSynchronize(stream_));
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
}

Device& device(int hip_device) {
  // The devices are leak intentionally as user code may still be accessing
  // device after main thread teardown.
  static auto* devices = []() {
    auto* devices = new std::vector<Device>;
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      devices->emplace_back(i);
    }
    return devices;
  }();
  return devices->at(hip_device);
}

Device& device(mlx::core::Device d) {
  return device(d.index);
}

CommandEncoder& get_command_encoder(Stream s) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    throw std::runtime_error(
        fmt::format("There is no Stream(gpu, {}) in current thread.", s.index));
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

} // namespace mlx::core::cu

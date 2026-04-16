#include "mlx/backend/rocm/rocm_utils.h"
// Copyright © 2026 Apple Inc.

#include "mlx/backend/rocm/quantized/qmm/qmm_impl_sm90.hip.h"

using namespace cute;

using TileShapeMN = Shape<_128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

QMM_SM90_GPU(TileShapeMN, ClusterShape)

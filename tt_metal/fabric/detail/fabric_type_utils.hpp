// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric::detail {

// A torus flag only changes connectivity when its dimension has a distinct
// wrap neighbor. On a size-two dimension, the wrap and ordinary neighbors are
// the same device, so that axis must retain mesh directionality.
inline FabricType collapse_torus_axes(FabricType fabric_type, const MeshShape& mesh_shape) {
    TT_FATAL(mesh_shape.dims() == 2, "Fabric type normalization requires a 2D mesh shape, got {}", mesh_shape);

    const bool has_distinct_x_wrap = has_flag(fabric_type, FabricType::TORUS_X) && mesh_shape[1] > 2;
    const bool has_distinct_y_wrap = has_flag(fabric_type, FabricType::TORUS_Y) && mesh_shape[0] > 2;
    if (has_distinct_x_wrap && has_distinct_y_wrap) {
        return FabricType::TORUS_XY;
    }
    if (has_distinct_x_wrap) {
        return FabricType::TORUS_X;
    }
    if (has_distinct_y_wrap) {
        return FabricType::TORUS_Y;
    }
    return FabricType::MESH;
}

}  // namespace tt::tt_fabric::detail

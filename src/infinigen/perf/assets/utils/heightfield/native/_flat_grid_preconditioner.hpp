// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2026 Princeton University

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "_quad_triangulation.hpp"

namespace infinigen::heightfield {

struct FlatGridResult {
  std::vector<uint32_t> faces;
  int flat_cells = 0;
  int flat_blocks = 0;
};

inline FlatGridResult precondition_flat_grid(
    const float *positions, int vertex_count, const int32_t *vertex_ids,
    int grid_height, int grid_width, const uint8_t *complete,
    float tolerance, int minimum_block, const float *metric_scale) {
  if (vertex_count < 0 || grid_height < 2 || grid_width < 2 ||
      tolerance < 0 || minimum_block < 2 ||
      (minimum_block & (minimum_block - 1)) != 0)
    throw std::invalid_argument("invalid flat-grid preconditioner input");
  const int cell_height = grid_height - 1;
  const int cell_width = grid_width - 1;
  FlatGridResult result;
  std::vector<uint8_t> claimed(size_t(cell_height) * cell_width, 0);

  auto vertex = [&](int row, int column) -> int32_t {
    return vertex_ids[size_t(row) * grid_width + column];
  };
  auto point = [&](int32_t index, int axis) -> double {
    return double(positions[size_t(index) * 3 + axis]) * metric_scale[axis];
  };
  auto block_is_flat = [&](int row, int column, int size) -> bool {
    for (int y = row; y < row + size; ++y)
      for (int x = column; x < column + size; ++x)
        if (!complete[size_t(y) * cell_width + x] ||
            claimed[size_t(y) * cell_width + x])
          return false;
    const int32_t ia = vertex(row, column);
    const int32_t ib = vertex(row, column + size);
    const int32_t ic = vertex(row + size, column);
    if (ia < 0 || ib < 0 || ic < 0) return false;
    const double ax = point(ia, 0), ay = point(ia, 1), az = point(ia, 2);
    const double ux = point(ib, 0) - ax, uy = point(ib, 1) - ay,
                 uz = point(ib, 2) - az;
    const double vx = point(ic, 0) - ax, vy = point(ic, 1) - ay,
                 vz = point(ic, 2) - az;
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double normal_length = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (!(normal_length > 0)) return false;
    const double limit = double(tolerance) * normal_length;
    for (int y = row; y <= row + size; ++y)
      for (int x = column; x <= column + size; ++x) {
        const int32_t index = vertex(y, x);
        if (index < 0) return false;
        const double distance = std::fabs(
            nx * (point(index, 0) - ax) + ny * (point(index, 1) - ay) +
            nz * (point(index, 2) - az));
        if (distance > limit) return false;
      }
    return true;
  };
  auto emit_flat_block = [&](int row, int column, int size) {
    auto emit = [&](uint32_t a, uint32_t b, uint32_t c) {
      result.faces.push_back(a); result.faces.push_back(b); result.faces.push_back(c);
    };
    // Keep every boundary lattice edge, and spread the transition over a coarse
    // inner grid rather than around one high-valence center vertex.
    const int step = std::min(8, std::max(2, size / 4));
    const uint32_t top_left_inner = uint32_t(vertex(row + step, column + step));
    const uint32_t top_right_inner = uint32_t(vertex(row + step, column + size - step));
    const uint32_t bottom_left_inner = uint32_t(vertex(row + size - step, column + step));
    const uint32_t bottom_right_inner = uint32_t(vertex(row + size - step, column + size - step));
    for (int offset = 0; offset < step; ++offset) {
      emit(uint32_t(vertex(row, column + offset)),
           uint32_t(vertex(row, column + offset + 1)), top_left_inner);
      emit(uint32_t(vertex(row + offset + 1, column)),
           uint32_t(vertex(row + offset, column)), top_left_inner);
      emit(uint32_t(vertex(row, column + size - step + offset)),
           uint32_t(vertex(row, column + size - step + offset + 1)), top_right_inner);
      emit(uint32_t(vertex(row + offset, column + size)),
           uint32_t(vertex(row + offset + 1, column + size)), top_right_inner);
      emit(bottom_left_inner,
           uint32_t(vertex(row + size, column + offset + 1)),
           uint32_t(vertex(row + size, column + offset)));
      emit(uint32_t(vertex(row + size - step + offset, column)), bottom_left_inner,
           uint32_t(vertex(row + size - step + offset + 1, column)));
      emit(bottom_right_inner,
           uint32_t(vertex(row + size, column + size - step + offset + 1)),
           uint32_t(vertex(row + size, column + size - step + offset)));
      emit(bottom_right_inner,
           uint32_t(vertex(row + size - step + offset, column + size)),
           uint32_t(vertex(row + size - step + offset + 1, column + size)));
    }
    for (int x = column + step; x < column + size - step; x += step) {
      const uint32_t inner_left = uint32_t(vertex(row + step, x));
      for (int dx = 0; dx < step; ++dx)
        emit(uint32_t(vertex(row, x + dx)),
             uint32_t(vertex(row, x + dx + 1)), inner_left);
      emit(uint32_t(vertex(row, x + step)),
           uint32_t(vertex(row + step, x + step)), inner_left);

      const uint32_t bottom_inner_left = uint32_t(vertex(row + size - step, x));
      emit(bottom_inner_left,
           uint32_t(vertex(row + size - step, x + step)),
           uint32_t(vertex(row + size, x + step)));
      for (int dx = step - 1; dx >= 0; --dx)
        emit(bottom_inner_left, uint32_t(vertex(row + size, x + dx + 1)),
             uint32_t(vertex(row + size, x + dx)));
    }
    for (int y = row + step; y < row + size - step; y += step) {
      const uint32_t left_inner_top = uint32_t(vertex(y, column + step));
      for (int dy = 0; dy < step; ++dy)
        emit(uint32_t(vertex(y + dy, column)), left_inner_top,
             uint32_t(vertex(y + dy + 1, column)));
      emit(uint32_t(vertex(y + step, column)), left_inner_top,
           uint32_t(vertex(y + step, column + step)));

      const uint32_t right_inner_top = uint32_t(vertex(y, column + size - step));
      for (int dy = 0; dy < step; ++dy)
        emit(right_inner_top, uint32_t(vertex(y + dy, column + size)),
             uint32_t(vertex(y + dy + 1, column + size)));
      emit(right_inner_top, uint32_t(vertex(y + step, column + size)),
           uint32_t(vertex(y + step, column + size - step)));
    }
    for (int y = row + step; y < row + size - step; y += step)
      for (int x = column + step; x < column + size - step; x += step) {
        const uint32_t a = uint32_t(vertex(y, x));
        const uint32_t b = uint32_t(vertex(y, x + step));
        const uint32_t c = uint32_t(vertex(y + step, x + step));
        const uint32_t d = uint32_t(vertex(y + step, x));
        emit(a, b, c); emit(a, c, d);
      }
    for (int y = row; y < row + size; ++y)
      for (int x = column; x < column + size; ++x)
        claimed[size_t(y) * cell_width + x] = 1;
    result.flat_cells += size * size;
    result.flat_blocks += 1;
  };

  int largest = 1;
  while (largest <= std::min(cell_height, cell_width) / 2) largest *= 2;
  for (int size = largest; size >= minimum_block; size /= 2)
    for (int row = 0; row + size <= cell_height; row += size)
      for (int column = 0; column + size <= cell_width; column += size)
        if (block_is_flat(row, column, size))
          emit_flat_block(row, column, size);

  std::vector<uint32_t> quads;
  for (int row = 0; row < cell_height; ++row)
    for (int column = 0; column < cell_width; ++column) {
      const size_t cell = size_t(row) * cell_width + column;
      if (!complete[cell] || claimed[cell]) continue;
      quads.push_back(uint32_t(vertex(row, column)));
      quads.push_back(uint32_t(vertex(row, column + 1)));
      quads.push_back(uint32_t(vertex(row + 1, column + 1)));
      quads.push_back(uint32_t(vertex(row + 1, column)));
    }
  if (!quads.empty()) {
    const size_t old = result.faces.size();
    result.faces.resize(old + quads.size() / 2 * 3);
    triangulate_quads_beauty(positions, quads.data(), quads.size() / 4,
                             result.faces.data() + old);
  }
  return result;
}

}  // namespace infinigen::heightfield

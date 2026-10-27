// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace mushroom_meshing
{
struct ProfilePlan
{
    std::vector<double> rs, zs;
    std::vector<int> offsets, sizes;
    size_t vertices = 0;
    int64_t triangles = 0;
};
inline ProfilePlan plan_profile(const float *input, int n, double step)
{
    if (n < 2 || !std::isfinite(step) || step <= 0)
        throw std::invalid_argument("Invalid profile");
    constexpr double pi = 3.14159265358979323846;
    ProfilePlan plan;
    auto &rs = plan.rs;
    auto &zs = plan.zs;
    for (int i = 0; i < n - 1; i++)
    {
        double r = input[3 * i], z = input[3 * i + 2], dr = input[3 * i + 3] - r,
               dz = input[3 * i + 5] - z;
        if (std::hypot(dr, dz) < 1e-8)
            continue;
        double raw_cuts = std::ceil(std::hypot(dr, dz) / step);
        if (raw_cuts > 100000)
            throw std::invalid_argument("Profile segment too large");
        int cuts = std::max(1, int(raw_cuts));
        for (int j = 0; j < cuts; j++)
        {
            double t = double(j) / cuts;
            rs.push_back(r + t * dr);
            zs.push_back(z + t * dz);
        }
    }
    rs.push_back(input[3 * (n - 1)]);
    zs.push_back(input[3 * (n - 1) + 2]);
    auto &offsets = plan.offsets;
    auto &sizes = plan.sizes;
    for (size_t i = 0; i < rs.size(); i++)
    {
        double r = rs[i], raw_ring = std::ceil(2 * pi * std::abs(r) / step / 4);
        if (raw_ring > 25000)
            throw std::invalid_argument("Profile radius too large");
        int m = std::abs(r) < 1e-8 ? 1 : std::max(4, 4 * int(raw_ring));
        if (m > 100000 || plan.vertices + size_t(m) > 20000000)
            throw std::invalid_argument("Mesh budget exceeded");
        offsets.push_back(plan.vertices);
        sizes.push_back(m);
        plan.vertices += m;
        if (i > 0)
            plan.triangles += (sizes[i - 1] > 1 ? sizes[i - 1] : 0) + (m > 1 ? m : 0);
    }
    return plan;
}
inline int64_t triangle_count(const float *input, int n, double step)
{
    return plan_profile(input, n, step).triangles;
}
inline void revolve(const float *input, int n, double step, std::vector<float> &co,
                    std::vector<int32_t> &loops, std::vector<int32_t> &counts,
                    std::vector<int32_t> *edges = nullptr,
                    std::vector<int32_t> *corner_edges = nullptr)
{
    constexpr double pi = 3.14159265358979323846;
    const auto plan = plan_profile(input, n, step);
    const auto &rs = plan.rs;
    const auto &zs = plan.zs;
    const auto &sizes = plan.sizes;
    const auto &offsets = plan.offsets;
    co.reserve(3 * plan.vertices);
    loops.reserve(3 * plan.triangles);
    counts.reserve(plan.triangles);
    for (size_t i = 0; i < rs.size(); i++)
    {
        double r = rs[i];
        int m = sizes[i];
        for (int j = 0; j < m; j++)
        {
            double a = 2 * pi * j / m;
            co.push_back(r * std::cos(a));
            co.push_back(r * std::sin(a));
            co.push_back(zs[i]);
        }
    }
    std::vector<int> ring_edges(sizes.size(), 0);
    auto edge = [&](int a, int b) {
        if (edges)
        {
            if (a > b)
                std::swap(a, b);
            edges->insert(edges->end(), {a, b});
        }
    };
    if (edges)
    {
        edges->reserve(3 * plan.vertices);
        corner_edges->reserve(3 * plan.triangles);
        if (sizes.size() > 1)
            for (size_t k = 0; k < sizes.size(); k++)
            {
                ring_edges[k] = edges->size() / 2;
                if (sizes[k] > 1)
                    for (int j = 0; j < sizes[k]; j++)
                        edge(offsets[k] + j, offsets[k] + (j + 1) % sizes[k]);
            }
    }
    auto tri = [&](int a, int b, int c, int e0 = 0, int e1 = 0, int e2 = 0) {
        loops.insert(loops.end(), {a, b, c});
        counts.push_back(3);
        if (corner_edges)
            corner_edges->insert(corner_edges->end(), {e0, e1, e2});
    };
    for (size_t k = 0; k + 1 < sizes.size(); k++)
    {
        int na = sizes[k], nb = sizes[k + 1], oa = offsets[k], ob = offsets[k + 1];
        if (na == 1 && nb == 1)
            continue;
        int bridge_begin = edges ? edges->size() / 2 : 0;
        if (na == 1)
        {
            for (int j = 0; j < nb; j++)
            {
                edge(oa, ob + j);
                tri(oa, ob + j, ob + (j + 1) % nb, bridge_begin + j, ring_edges[k + 1] + j,
                    bridge_begin + (j + 1) % nb);
            }
            continue;
        }
        if (nb == 1)
        {
            for (int i = 0; i < na; i++)
            {
                edge(oa + i, ob);
                tri(oa + i, ob, oa + (i + 1) % na, bridge_begin + i, bridge_begin + (i + 1) % na,
                    ring_edges[k] + i);
            }
            continue;
        }
        int i = 0, j = 0, face = 0, bridge_count = na + nb - std::gcd(na, nb);
        while (i < na || j < nb)
        {
            int64_t aa = int64_t(i + 1) * nb, bb = int64_t(j + 1) * na;
            int a = oa + i % na, b = ob + j % nb;
            edge(a, b);
            int current = bridge_begin + face, next = bridge_begin + (face + 1) % bridge_count;
            if (aa == bb)
            {
                loops.insert(loops.end(), {a, b, ob + (j + 1) % nb, oa + (i + 1) % na});
                counts.push_back(4);
                if (corner_edges)
                    corner_edges->insert(corner_edges->end(), {current, ring_edges[k + 1] + j % nb,
                                                               next, ring_edges[k] + i % na});
                i++;
                j++;
            }
            else if (aa < bb)
            {
                tri(a, b, oa + (i + 1) % na, current, next, ring_edges[k] + i % na);
                i++;
            }
            else
            {
                tri(a, b, ob + (j + 1) % nb, current, ring_edges[k + 1] + j % nb, next);
                j++;
            }
            face++;
        }
    }
    // Choose outward winding for closed profiles; no point positions are changed.
    double volume = 0;
    size_t at = 0;
    for (int m : counts)
    {
        int a = loops[at];
        for (int j = 1; j < m - 1; j++)
        {
            int b = loops[at + j], c = loops[at + j + 1];
            volume += double(co[3 * a]) * (double(co[3 * b + 1]) * co[3 * c + 2] -
                                           double(co[3 * b + 2]) * co[3 * c + 1]) +
                      double(co[3 * a + 1]) *
                          (double(co[3 * b + 2]) * co[3 * c] - double(co[3 * b]) * co[3 * c + 2]) +
                      double(co[3 * a + 2]) *
                          (double(co[3 * b]) * co[3 * c + 1] - double(co[3 * b + 1]) * co[3 * c]);
        }
        at += m;
    }
    if (volume < 0)
    {
        at = 0;
        for (int m : counts)
        {
            std::reverse(loops.begin() + at, loops.begin() + at + m);
            if (corner_edges)
                std::reverse(corner_edges->begin() + at, corner_edges->begin() + at + m - 1);
            at += m;
        }
    }
}

inline void revolve_edges(const float *input, int n, double step, std::vector<float> &co,
                          std::vector<int32_t> &loops, std::vector<int32_t> &counts,
                          std::vector<int32_t> &edges, std::vector<int32_t> &corners)
{
    revolve(input, n, step, co, loops, counts, &edges, &corners);
}
inline void mesh_edges(const int32_t *loops, const int32_t *counts, int nf,
                       std::vector<int32_t> &edges, std::vector<int32_t> &corner_edges)
{
    size_t nl = 0;
    for (int i = 0; i < nf; i++)
        nl += counts[i];
    size_t capacity = 1;
    while (capacity < nl * 2)
        capacity <<= 1;
    std::vector<uint64_t> keys(capacity, 0);
    std::vector<int32_t> values(capacity, 0);
    corner_edges.resize(nl);
    edges.reserve(nl * 2);
    size_t offset = 0;
    for (int f = 0; f < nf; f++)
    {
        int n = counts[f];
        for (int j = 0; j < n; j++)
        {
            int32_t a = loops[offset + j], b = loops[offset + (j + 1) % n];
            if (a > b)
                std::swap(a, b);
            uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
            uint64_t hash = key;
            hash ^= hash >> 33;
            hash *= 0xff51afd7ed558ccdULL;
            hash ^= hash >> 33;
            size_t slot = hash & (capacity - 1);
            while (keys[slot] && keys[slot] != key + 1)
                slot = (slot + 1) & (capacity - 1);
            if (!keys[slot])
            {
                keys[slot] = key + 1;
                values[slot] = edges.size() / 2;
                edges.push_back(a);
                edges.push_back(b);
            }
            corner_edges[offset + j] = values[slot];
        }
        offset += n;
    }
}
} // namespace mushroom_meshing

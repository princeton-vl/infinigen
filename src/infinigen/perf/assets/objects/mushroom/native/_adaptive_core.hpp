#define TIGHT_BOUNDS 1
// SPDX-License-Identifier: BSD-3-Clause
// Lazy surface-voxel hierarchy for clearance queries between mushrooms.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>
#include <string>
#include <vector>
using V = std::array<double, 3>;
V operator+(V a, V b)
{
    for (int k = 0; k < 3; k++)
        a[k] += b[k];
    return a;
}
V operator-(V a, V b)
{
    for (int k = 0; k < 3; k++)
        a[k] -= b[k];
    return a;
}
V operator*(V a, double b)
{
    for (auto &x : a)
        x *= b;
    return a;
}
double dot(V a, V b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
V cross(V a, V b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm2(V a)
{
    return dot(a, a);
}
struct Box
{
    V lo, hi;
};
struct Tri
{
    V a, b, c;
    Box box;
};
Box bounds(V a, V b, V c)
{
    Box box;
    for (int k = 0; k < 3; k++)
    {
        box.lo[k] = std::min({a[k], b[k], c[k]});
        box.hi[k] = std::max({a[k], b[k], c[k]});
    }
    return box;
}
bool overlap(Box a, Box b)
{
    for (int k = 0; k < 3; k++)
        if (a.lo[k] > b.hi[k] || b.lo[k] > a.hi[k])
            return false;
    return true;
}
double gap2(Box a, Box b, V off)
{
    double s = 0;
    for (int k = 0; k < 3; k++)
    {
        double d = std::max({0., a.lo[k] - b.hi[k] - off[k], b.lo[k] + off[k] - a.hi[k]});
        s += d * d;
    }
    return s;
}
double segment2(V p, V q, V a, V b)
{
    V d1 = q - p, d2 = b - a, r = p - a;
    double x = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), s = 0, t = 0;
    const double eps = 1e-28;
    if (x <= eps && e <= eps)
        return norm2(r);
    if (x <= eps)
        t = std::clamp(f / e, 0., 1.);
    else
    {
        double c = dot(d1, r);
        if (e <= eps)
            s = std::clamp(-c / x, 0., 1.);
        else
        {
            double h = dot(d1, d2), den = x * e - h * h;
            if (den > eps)
                s = std::clamp((h * f - c * e) / den, 0., 1.);
            t = (h * s + f) / e;
            if (t < 0)
            {
                t = 0;
                s = std::clamp(-c / x, 0., 1.);
            }
            else if (t > 1)
            {
                t = 1;
                s = std::clamp((h - c) / x, 0., 1.);
            }
        }
    }
    return norm2((p + d1 * s) - (a + d2 * t));
}
double point2(V p, Tri t)
{
    V ab = t.b - t.a, ac = t.c - t.a, ap = p - t.a;
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0)
        return norm2(ap);
    V bp = p - t.b;
    double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3)
        return norm2(bp);
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && d1 != d3)
        return norm2(p - (t.a + ab * (d1 / (d1 - d3))));
    V cp = p - t.c;
    double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6)
        return norm2(cp);
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0 && d2 != d6)
        return norm2(p - (t.a + ac * (d2 / (d2 - d6))));
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 && (d4 - d3 + d5 - d6) != 0)
        return norm2(p - (t.b + (t.c - t.b) * ((d4 - d3) / (d4 - d3 + d5 - d6))));
    double den = va + vb + vc;
    if (std::abs(den) < 1e-28)
        return std::min(
            {segment2(p, p, t.a, t.b), segment2(p, p, t.b, t.c), segment2(p, p, t.c, t.a)});
    return norm2(p - (t.a + ab * (vb / den) + ac * (vc / den)));
}
bool hits(V p, V q, Tri t)
{
    V d = q - p, e1 = t.b - t.a, e2 = t.c - t.a, h = cross(d, e2);
    double det = dot(e1, h);
    if (std::abs(det) < 1e-20)
        return false;
    V s = p - t.a;
    double u = dot(s, h) / det;
    if (u < 0 || u > 1)
        return false;
    V v = cross(s, e1);
    double w = dot(d, v) / det;
    if (w < 0 || u + w > 1)
        return false;
    double f = dot(e2, v) / det;
    return f >= 0 && f <= 1;
}
double distance2(Tri a, Tri b)
{
    V av[3] = {a.a, a.b, a.c}, bv[3] = {b.a, b.b, b.c};
    double best = 1e300;
    for (int i = 0; i < 3; i++)
    {
        if (hits(av[i], av[(i + 1) % 3], b) || hits(bv[i], bv[(i + 1) % 3], a))
            return 0;
        best = std::min({best, point2(av[i], b), point2(bv[i], a)});
        for (int j = 0; j < 3; j++)
            best = std::min(best, segment2(av[i], av[(i + 1) % 3], bv[j], bv[(j + 1) % 3]));
    }
    return best;
}
#ifndef LEAF_SIZE
#define LEAF_SIZE 48
#endif
struct Node
{
    Box box;
    std::vector<int> ids;
    std::vector<std::unique_ptr<Node>> children;
    int depth = 0;
};
struct Tree
{
    bool use_bvh =
        std::string(std::getenv("INFINIGEN_MUSHROOM_TREE") ? std::getenv("INFINIGEN_MUSHROOM_TREE")
                                                           : "bvh") == "bvh";

    std::vector<Tri> tris;
    std::unique_ptr<Node> root;
    int nodes = 0, splits = 0;
    double mincell;
    long tests = 0, pairs = 0;
    explicit Tree(double cell) : mincell(cell)
    {
        root = std::make_unique<Node>();
        root->box = {{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
        nodes = 1;
    }
    void add(V a, V b, V c)
    {
        Tri t{a, b, c, bounds(a, b, c)};
        root->ids.push_back(tris.size());
        tris.push_back(t);
        for (int k = 0; k < 3; k++)
        {
            root->box.lo[k] = std::min(root->box.lo[k], t.box.lo[k]);
            root->box.hi[k] = std::max(root->box.hi[k], t.box.hi[k]);
        }
    }
    Tree(const double *xyz, int n, double cell) : Tree(cell)
    {
        tris.reserve(n);
        root->ids.reserve(n);
        for (int i = 0; i < n; i++)
        {
            V a, b, c;
            for (int k = 0; k < 3; k++)
            {
                a[k] = xyz[i * 9 + k];
                b[k] = xyz[i * 9 + 3 + k];
                c[k] = xyz[i * 9 + 6 + k];
            }
            add(a, b, c);
        }
    }
    Tree(const float *xyz, const int32_t *ids, int n, const double *mat, double height, double cell)
        : Tree(cell)
    {
        tris.reserve(n);
        root->ids.reserve(n);
        int count_vertices = 0;
        for (int i = 0; i < n * 3; i++)
            count_vertices = std::max(count_vertices, ids[i] + 1);
        std::vector<V> transformed(count_vertices);
        std::vector<uint8_t> ready(count_vertices, 0);
        auto transform_vertex = [&](int id) {
            if (!ready[id])
            {
                for (int k = 0; k < 3; k++)
                    transformed[id][k] = mat[k * 3] * xyz[3 * id] +
                                         mat[k * 3 + 1] * xyz[3 * id + 1] +
                                         mat[k * 3 + 2] * xyz[3 * id + 2];
                ready[id] = 1;
            }
            return transformed[id];
        };
        for (int i = 0; i < n; i++)
        {
            const int a = ids[3 * i], b = ids[3 * i + 1], c = ids[3 * i + 2];
            const bool ia = xyz[3 * a + 2] >= height, ib = xyz[3 * b + 2] >= height,
                       ic = xyz[3 * c + 2] >= height;
            if (!ia && !ib && !ic)
                continue;
            if (ia && ib && ic)
            {
                add(transform_vertex(a), transform_vertex(b), transform_vertex(c));
                continue;
            }
            V v[3], poly[6];
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    v[j][k] = xyz[3 * ids[3 * i + j] + k];
            int count = 0;
            for (int j = 0; j < 3; j++)
            {
                V a = v[j], b = v[(j + 1) % 3];
                bool ia = a[2] >= height, ib = b[2] >= height;
                if (ia)
                    poly[count++] = a;
                if (ia != ib)
                    poly[count++] = a + (b - a) * ((height - a[2]) / (b[2] - a[2]));
            }
            for (int j = 0; j < count; j++)
            {
                V a = poly[j];
                for (int k = 0; k < 3; k++)
                    poly[j][k] = mat[k * 3] * a[0] + mat[k * 3 + 1] * a[1] + mat[k * 3 + 2] * a[2];
            }
            for (int j = 1; j + 1 < count; j++)
                add(poly[0], poly[j], poly[j + 1]);
        }
    }

    bool leaf(Node *p)
    {
        double side = 0;
        for (int k = 0; k < 3; k++)
            side = std::max(side, p->box.hi[k] - p->box.lo[k]);
        return p->ids.size() <= LEAF_SIZE || side <= mincell || p->depth >= (use_bvh ? 24 : 12);
    }
    void split(Node *p)
    {
        if (!p->children.empty())
            return;
        if (use_bvh)
        {
            int axis = 0;
            for (int k = 1; k < 3; k++)
                if (p->box.hi[k] - p->box.lo[k] > p->box.hi[axis] - p->box.lo[axis])
                    axis = k;
            const size_t mid_index = p->ids.size() / 2;
            std::nth_element(p->ids.begin(), p->ids.begin() + mid_index, p->ids.end(),
                             [&](int a, int b) {
                                 double ca = tris[a].box.lo[axis] + tris[a].box.hi[axis];
                                 double cb = tris[b].box.lo[axis] + tris[b].box.hi[axis];
                                 return ca == cb ? a < b : ca < cb;
                             });
            for (int side = 0; side < 2; side++)
            {
                auto c = std::make_unique<Node>();
                c->depth = p->depth + 1;
                c->box = {{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
                size_t begin = side ? mid_index : 0, end = side ? p->ids.size() : mid_index;
                c->ids.assign(p->ids.begin() + begin, p->ids.begin() + end);
                for (int id : c->ids)
                    for (int k = 0; k < 3; k++)
                    {
                        c->box.lo[k] = std::min(c->box.lo[k], tris[id].box.lo[k]);
                        c->box.hi[k] = std::max(c->box.hi[k], tris[id].box.hi[k]);
                    }
                p->children.push_back(std::move(c));
                nodes++;
            }
            splits++;
            return;
        }
        V mid = (p->box.lo + p->box.hi) * .5;
        for (int mask = 0; mask < 8; mask++)
        {
            auto c = std::make_unique<Node>();
            c->depth = p->depth + 1;
            for (int k = 0; k < 3; k++)
            {
                c->box.lo[k] = (mask & (1 << k)) ? mid[k] : p->box.lo[k];
                c->box.hi[k] = (mask & (1 << k)) ? p->box.hi[k] : mid[k];
            }
#ifdef TIGHT_BOUNDS
            Box tight{{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
#endif
            for (int id : p->ids)
                if (overlap(tris[id].box, c->box))
                {
                    c->ids.push_back(id);
#ifdef TIGHT_BOUNDS
                    for (int k = 0; k < 3; k++)
                    {
                        tight.lo[k] = std::min(tight.lo[k], tris[id].box.lo[k]);
                        tight.hi[k] = std::max(tight.hi[k], tris[id].box.hi[k]);
                    }
#endif
                }
#ifdef TIGHT_BOUNDS
            for (int k = 0; k < 3; k++)
            {
                c->box.lo[k] = std::max(c->box.lo[k], tight.lo[k]);
                c->box.hi[k] = std::min(c->box.hi[k], tight.hi[k]);
            }
#endif
            if (!c->ids.empty())
            {
                p->children.push_back(std::move(c));
                nodes++;
            }
        }
        splits++;
    }
};
struct Pair
{
    Node *a;
    Node *b;
    double d;
    bool operator<(const Pair &o) const
    {
        if (d != o.d)
            return d > o.d;
        return a->depth + b->depth < o.a->depth + o.b->depth;
    }
};
bool plane_far(const Tri &a, const Tri &b, double lim)
{
    V n = cross(a.b - a.a, a.c - a.a);
    double x = dot(n, b.a - a.a), y = dot(n, b.b - a.a), z = dot(n, b.c - a.a);
    double gap = std::max(std::min({x, y, z}), -std::max({x, y, z}));
    // Conservative guard around the threshold; degenerate planes never reject.
    return gap > 0 && gap * gap > (lim + 1e-16) * norm2(n) * (1 + 1e-12);
}
bool clear(Tree &a, Tree &b, V off, double distance)
{
    if (a.tris.empty() || b.tris.empty())
        return true;
    double lim = distance * distance;
    std::priority_queue<Pair> todo;
    todo.push({a.root.get(), b.root.get(), gap2(a.root->box, b.root->box, off)});
    while (!todo.empty())
    {
        auto p = todo.top();
        todo.pop();
        a.pairs++;
        if (p.d > lim)
            continue;
        bool la = a.leaf(p.a), lb = b.leaf(p.b);
        if (la && lb)
        {
            for (int ai : p.a->ids)
                for (int bi : p.b->ids)
                {
                    Tri x = a.tris[ai], y = b.tris[bi];
                    if (gap2(x.box, y.box, off) > lim)
                        continue;
                    y.a = y.a + off;
                    y.b = y.b + off;
                    y.c = y.c + off;
                    if (plane_far(x, y, lim) || plane_far(y, x, lim))
                        continue;
                    a.tests++;
                    if (distance2(x, y) <= lim)
                        return false;
                }
        }
        else
        {
            bool split_a = !la && (lb || p.a->ids.size() >= p.b->ids.size());
            Tree &t = split_a ? a : b;
            Node *n = split_a ? p.a : p.b;
            t.split(n);
            for (auto &c : n->children)
            {
                Node *x = split_a ? c.get() : p.a, *y = split_a ? p.b : c.get();
                double d = gap2(x->box, y->box, off);
                if (d <= lim)
                    todo.push({x, y, d});
            }
        }
    }
    return true;
}
extern "C"
{
    void *av_create(const double *x, int n, double c)
    {
        try
        {
            return new Tree(x, n, c);
        }
        catch (...)
        {
            return nullptr;
        }
    }
    void *av_create_mesh(const float *x, const int32_t *ids, int n, const double *m, double h,
                         double c)
    {
        try
        {
            return new Tree(x, ids, n, m, h, c);
        }
        catch (...)
        {
            return nullptr;
        }
    }
    void av_free(void *t)
    {
        delete static_cast<Tree *>(t);
    }
    int av_clear(void *a, void *b, const double *o, double d)
    {
        try
        {
            return clear(*static_cast<Tree *>(a), *static_cast<Tree *>(b), {o[0], o[1], o[2]}, d);
        }
        catch (...)
        {
            return -1;
        }
    }
    void av_stats(void *p, long *out)
    {
        Tree &t = *static_cast<Tree *>(p);
        out[0] = t.nodes;
        out[1] = t.splits;
        out[2] = t.tests;
        out[3] = t.pairs;
        out[4] = t.tris.size();
    }
    double av_distance(const double *a, const double *b)
    {
        Tri x, y;
        for (int k = 0; k < 3; k++)
        {
            x.a[k] = a[k];
            x.b[k] = a[k + 3];
            x.c[k] = a[k + 6];
            y.a[k] = b[k];
            y.b[k] = b[k + 3];
            y.c[k] = b[k + 6];
        }
        return std::sqrt(distance2(x, y));
    }
}

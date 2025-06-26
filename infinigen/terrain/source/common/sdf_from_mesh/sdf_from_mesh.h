// Copyright (C) 2024, Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma

DEVICE_FUNC void sdf_from_mesh_common(
    float3_nonbuiltin position,
    float *sdf,
    float3_nonbuiltin *vertices,
    int *faces,
    int n_faces
) {
    const float eps = 1e-5, eps3=0;
    float min_dist=1e9;
    float abs_plane_dist;
    int sign;
    for (int i = 0; i < n_faces; i++) {
        float3_nonbuiltin verts[3];
        verts[0] = vertices[faces[i * 3]];
        verts[1] = vertices[faces[i * 3 + 1]];
        verts[2] = vertices[faces[i * 3 + 2]];
        float3_nonbuiltin v12 = verts[2] - verts[1];
        float v12n = v12.norm();
        if (v12n < eps) continue;
        float3_nonbuiltin v01 = verts[1] - verts[0], v02 = verts[2] - verts[0];
        float v01n = v01.norm(), v02n = v02.norm();
        if (v01n < eps || v02n < eps) continue;
        float3_nonbuiltin n = v01.cross(v02);
        float plane_dist = (position - verts[0]).dot(n);
        float n1 = n.norm();
        float n2 = n1 * n1;
        if (n1 < eps) continue;
        float3_nonbuiltin projected = position - n * plane_dist / n2;
        plane_dist /= n1;
        float p01 = v01.cross(projected - verts[0]).dot(n) / n2;
        float p02 = -(v02.cross(projected - verts[0]).dot(n)) / n2;
        float p12 = 1 - p01 - p02;
        float dist;
        if (p01 >= -eps && p02 >= -eps && p12 >= -eps) {
            dist = abs(plane_dist);
        }
        else {
            dist = 1e9;
            for (int j = 0; j < 3; j++) {
                dist = min(dist, (projected - verts[j]).norm());
            }
            float k = (projected - verts[0]).dot(v01), edge_dist;
            if (k >= 0 && k <= v01n*v01n) {
                float3_nonbuiltin v0p = projected - verts[0];
                edge_dist = v0p.cross(v01).norm() / v01n;
                dist = min(dist, edge_dist);
            }
            k = (projected - verts[0]).dot(v02);
            if (k >= 0 && k <= v02n*v02n) {
                float3_nonbuiltin v0p = projected - verts[0];
                edge_dist = v0p.cross(v02).norm() / v02n;
                dist = min(dist, edge_dist);
            }
            k = (projected - verts[1]).dot(v12);
            if (k >= 0 && k <= v12n*v12n) {
                float3_nonbuiltin v1p = projected - verts[1];
                edge_dist = v1p.cross(v12).norm() / v12n;
                dist = min(dist, edge_dist);
            }
            dist = sqrt(dist*dist + plane_dist*plane_dist);
        }
        if (dist < min_dist - eps) {
            min_dist = dist;
            abs_plane_dist = abs(plane_dist);
            sign = plane_dist >= -eps3;
        }
        else {
            if (dist < min_dist + eps) {
                if (abs(plane_dist) > abs_plane_dist) {
                    abs_plane_dist = abs(plane_dist);
                    sign = plane_dist >= -eps3;
                }
            }
        }
    }
    assert(min_dist != 1e9);
    *sdf = sign? abs(min_dist): -abs(min_dist);
}
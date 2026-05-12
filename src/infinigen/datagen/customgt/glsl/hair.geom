#version 440 core

layout (lines_adjacency) in;
layout (triangle_strip, max_vertices = 60) out;

out vec3 interp_pos_cc; // technically opengl
out vec3 interp_pos_cc_next; // technically opengl
out vec3 pa;
out vec3 pb;
out float hair_width;

uniform mat4 projection;

in VS_OUT {
    vec3 pos_wc;
	vec3 pos_cc;
    vec3 pos_cc_next;
	float radius;
} gs_in[];

vec3 forward = vec3(0.0, 0.0, -1.0);

vec3 catmull_rom(float t, vec3 P0, vec3 P1, vec3 P2, vec3 P3){
    float t0 = 0;
    float t1 = length(P1 - P0) + t0;
    float t2 = length(P2 - P1) + t1;
    float t3 = length(P3 - P2) + t2;

    vec3 A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1;
    vec3 A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2;
    vec3 A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3;
    vec3 B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2;
    vec3 B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3;
    vec3 C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2;
    return C;
}

void save(vec3 v, vec3 v_next){
    gl_Position = projection * vec4(v, 1.0);
    interp_pos_cc = v;
    interp_pos_cc_next = v_next;
    EmitVertex();
}

vec3[4] hitbox_between_points(vec3 p1, vec3 p2){
    vec3 vs[4];
    vec3 perp = normalize(cross(p1 - p2, forward)) * 4e-3;
    vs[0] = p1 + perp;
    vs[1] = p1 - perp;
    vs[2] = p2 + perp;
    vs[3] = p2 - perp;
    return vs;
}

void main() {

    vec3 P0 = gs_in[0].pos_cc;
    vec3 P1 = gs_in[1].pos_cc;
    vec3 P2 = gs_in[2].pos_cc;
    vec3 P3 = gs_in[3].pos_cc;

    vec3 P0n = gs_in[0].pos_cc_next;
    vec3 P1n = gs_in[1].pos_cc_next;
    vec3 P2n = gs_in[2].pos_cc_next;
    vec3 P3n = gs_in[3].pos_cc_next;

    float t1 = length(P1 - P0);
    float t2 = length(P2 - P1) + t1;

    float r1 = gs_in[1].radius * 2;
    float r2 = gs_in[2].radius * 2;

    float step = 0.1;
    for (float f=0; f<1; f+=step){ // f in [0, 1]
        float rad1 = f        * (r2 - r1) + r1;
        float rad2 = (f+step) * (r2 - r1) + r1;

        // Current
        vec3 point1 = catmull_rom(f * (t2 - t1) + t1, P0, P1, P2, P3);
        vec3 point2 = catmull_rom((f+step) * (t2 - t1) + t1, P0, P1, P2, P3);
        vec3 hitbox_cc[4] = hitbox_between_points(point1, point2);
        pa = point1;
        pb = point2;

        // Next
        vec3 point1n = catmull_rom(f * (t2 - t1) + t1, P0n, P1n, P2n, P3n);
        vec3 point2n = catmull_rom((f+step) * (t2 - t1) + t1, P0n, P1n, P2n, P3n);
        vec3 hitbox_cc_next[4] = hitbox_between_points(point1n, point2n);

        hair_width = rad1;
        save(hitbox_cc[0], hitbox_cc_next[0]);
        hair_width = rad2;
        save(hitbox_cc[2], hitbox_cc_next[2]);
        save(hitbox_cc[3], hitbox_cc_next[3]);

        hair_width = rad1;
        save(hitbox_cc[0], hitbox_cc_next[0]);
        save(hitbox_cc[1], hitbox_cc_next[1]);
        hair_width = rad2;
        save(hitbox_cc[2], hitbox_cc_next[2]);
    }
}
#version 440 core

layout (lines_adjacency) in;
layout (line_strip, max_vertices = 2) out;

uniform mat4 cc2img;

in VS_OUT {
    vec3 pos_wc;
    vec3 pos_cc;
    vec3 pos_cc_next;
    bool has_flow;
    int vertex_id;
    ivec3 instance_id;
    int tag;
} gs_in[];

vec2 proj(vec3 v){
    vec4 h = cc2img * vec4(v, 1.0);
    return vec2(h[0] / abs(h[2]), h[1] / abs(h[2]));
}

vec4 offset_amount = vec4(0.0, 0.0, -0.00001, 0.00);

bool is_frontfacing(vec3 v1, vec3 v2, vec3 v3){
    vec2 uv1 = proj(v1);
    vec2 uv2 = proj(v2);
    vec2 uv3 = proj(v3);

    vec2 a = (uv2-uv1);
    vec2 b = (uv3-uv2);
    float winding = a.x * b.y - b.x * a.y;
    return winding > 0;

}

void main() {
    vec3 v1 = gs_in[0].pos_cc; // A
    vec3 v2 = gs_in[1].pos_cc; // B
    vec3 v3 = gs_in[2].pos_cc;
    vec3 v4 = gs_in[3].pos_cc;

    float z_min = 0.00001;
    vec3 v_pos = (v1.z > v2.z) ? v1 : v2;

    if (v1.z < z_min) {
        float t = (z_min - v1.z) / max(v_pos.z - v1.z, 1e-8);
        t = clamp(t, 0.0, 1.0);
        v1 = mix(v1, v_pos, t);
    }
    if (v2.z < z_min) {
        float t = (z_min - v2.z) / max(v_pos.z - v2.z, 1e-8);
        t = clamp(t, 0.0, 1.0);
        v2 = mix(v2, v_pos, t);
    }
    if (v3.z < z_min) {
        float t = (z_min - v3.z) / max(v_pos.z - v3.z, 1e-8);
        t = clamp(t, 0.0, 1.0);
        v3 = mix(v3, v_pos, t);
    }
    if (v4.z < z_min) {
        float t = (z_min - v4.z) / max(v_pos.z - v4.z, 1e-8);
        t = clamp(t, 0.0, 1.0);
        v4 = mix(v4, v_pos, t);
    }


    bool draw_boundary = (is_frontfacing(v1, v2, v3) != is_frontfacing(v2, v1, v4));

    if (draw_boundary){
        gl_Position = gl_in[0].gl_Position + offset_amount;
        EmitVertex();
        gl_Position = gl_in[1].gl_Position + offset_amount;
        EmitVertex();
    }
}

#version 440 core

layout (lines_adjacency) in;
layout (line_strip, max_vertices = 2) out;

uniform mat4 wc2img;

out vec3 normal;
out vec3 interp_pos_wc;

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
    vec4 h = wc2img * vec4(v, 1.0);
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
    vec3 v1 = gs_in[0].pos_wc; // A
    vec3 v2 = gs_in[1].pos_wc; // B
    vec3 v3 = gs_in[2].pos_wc;
    vec3 v4 = gs_in[3].pos_wc;

    bool draw_boundary = (is_frontfacing(v1, v2, v3) != is_frontfacing(v2, v1, v4));

    if (draw_boundary){
        gl_Position = gl_in[0].gl_Position + offset_amount;
        EmitVertex();
        gl_Position = gl_in[1].gl_Position + offset_amount;
        EmitVertex();
    }
}
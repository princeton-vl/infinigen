#version 440 core

layout (lines_adjacency) in;
layout (triangle_strip, max_vertices = 6) out;

out vec3 normal;
out vec3 cc_normal;
out vec3 interp_pos_wc;
out vec3 interp_pos_cc;
out vec3 interp_pos_cc_next;
out float tri_area;
out float px_area;
out float has_flow;
out ivec3 face_id;
out ivec3 instance_id;
out ivec3 tag;

uniform mat4 wc2img;

in VS_OUT {
    vec3 pos_wc;
    vec3 pos_cc;
    vec3 pos_cc_next;
    bool has_flow;
    int vertex_id;
    ivec3 instance_id;
    int tag;
} gs_in[];

vec2 proj(vec3 inp){
    vec4 tmp = vec4(inp, 1.0);
    vec4 tmp2 = wc2img * tmp;
    return vec2(tmp2[0]/tmp2[2], tmp2[1]/tmp2[2]);
}

bool is_frontfacing(vec3 v1, vec3 v2, vec3 v3){
    vec2 uv1 = proj(v1);
    vec2 uv2 = proj(v2);
    vec2 uv3 = proj(v3);

    vec2 a = (uv2-uv1);
    vec2 b = (uv3-uv2);
    float winding = a.x * b.y - b.x * a.y;
    return winding > 0;
}

vec3 to_z_B_y_U(vec3 v){
    return vec3(v.x, -v.y, -v.z);
}

void save(int i){
    gl_Position = gl_in[i].gl_Position;
    interp_pos_wc = gs_in[i].pos_wc;
    interp_pos_cc = gs_in[i].pos_cc;
    interp_pos_cc_next = gs_in[i].pos_cc_next;
    EmitVertex();
}

void main() {

    face_id = ivec3(gs_in[0].vertex_id, gs_in[1].vertex_id, gs_in[2].vertex_id);
    instance_id = gs_in[0].instance_id;

    has_flow = float(gs_in[0].has_flow && gs_in[1].has_flow && gs_in[2].has_flow && gs_in[3].has_flow);

    vec3 v1 = gs_in[0].pos_wc;
    vec3 v2 = gs_in[1].pos_wc;
    vec3 v3 = gs_in[2].pos_wc;
    vec3 v4 = gs_in[3].pos_wc;

    vec2 p1 = proj(v1);
    vec2 p2 = proj(v2);
    vec2 p3 = proj(v3);
    vec2 p4 = proj(v4);
    vec2 a = p1 - p2;
    vec2 b = p1 - p3;
    vec2 c = p1 - p4;

    // First triangle
    cc_normal = normalize(cross(to_z_B_y_U(gs_in[0].pos_cc-gs_in[1].pos_cc), to_z_B_y_U(gs_in[0].pos_cc-gs_in[2].pos_cc)));
    tri_area = length(cross(v1 - v2, v1 - v3))/2;
    normal = normalize(cross(v1-v2, v1-v3));
    px_area = abs(a.x*b.y - a.y*b.x)/2;
    tag = ivec3(gs_in[0].tag, gs_in[1].tag, gs_in[2].tag);

    if (!is_frontfacing(v1, v2, v3))
        cc_normal = -cc_normal;

    save(0);
    save(1);
    save(2);

    // Second triangle
    cc_normal = normalize(cross(to_z_B_y_U(gs_in[0].pos_cc-gs_in[1].pos_cc), to_z_B_y_U(gs_in[0].pos_cc-gs_in[3].pos_cc)));
    tri_area = length(cross(v1 - v2, v1 - v4))/2;
    normal = normalize(cross(v1-v2, v1-v4));
    px_area = abs(a.x*c.y - a.y*c.x)/2;
    tag = ivec3(gs_in[0].tag, gs_in[1].tag, gs_in[3].tag);

    if (!is_frontfacing(v1, v2, v4))
        cc_normal = -cc_normal;

    save(0);
    save(1);
    save(3);
}
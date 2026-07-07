// VERTEX SHADER

#version 440 core

layout (location = 0) in mat4 instanceMatrix;
layout (location = 4) in mat4 instanceMatrixNext;
layout (location = 8) in vec3 aPos;
layout (location = 9) in vec3 aPos_next;
layout (location = 10) in ivec3 instance_id;
layout (location = 11) in int tag;


uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewNext;

mat4 opengl_to_cv = mat4(
	vec4(1., 0., 0., 0.),
	vec4(0., -1., 0., 0.),
	vec4(0., 0., -1., 0.),
	vec4(0., 0., 0., 1.));

out VS_OUT {
    vec3 pos_wc;
	vec3 pos_cc;
	vec3 pos_cc_next;
	bool has_flow;
	int vertex_id;
	ivec3 instance_id;
	int tag;
} vs_out;

void main() {
	vec4 pos_wc = instanceMatrix * vec4(aPos, 1.0);
	vs_out.pos_wc = vec3(pos_wc);

	vec4 pos_cc_opengl = view * pos_wc;
	vec4 pos_cc = opengl_to_cv * pos_cc_opengl;
	vs_out.pos_cc = pos_cc.xyz / pos_cc.w;

	vs_out.has_flow = (abs(instanceMatrixNext[3][3]) > 1e-4);

	vec4 pos_wc_next = instanceMatrixNext * vec4(aPos_next, 1.0);
	vec4 pos_cc_next_opengl = viewNext * pos_wc_next;
	vec4 pos_cc_next = opengl_to_cv * pos_cc_next_opengl;
	vs_out.pos_cc_next = pos_cc_next.xyz / pos_cc_next.w;

	gl_Position = projection * pos_cc_next_opengl;

	vs_out.vertex_id = gl_VertexID;
	vs_out.instance_id = instance_id;

	vs_out.tag = tag;
}

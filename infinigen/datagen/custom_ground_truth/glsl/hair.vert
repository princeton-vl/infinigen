// VERTEX SHADER

#version 440 core

layout (location = 0) in mat4 instanceMatrix;
layout (location = 4) in mat4 instanceMatrixNext;
layout (location = 8) in vec3 aPos;
layout (location = 9) in vec3 aPos_next;
layout (location = 10) in ivec3 instance_id;
layout (location = 11) in int tag;
layout (location = 12) in float radius;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewNext;

out VS_OUT {
    vec3 pos_wc;
	vec3 pos_cc;
	vec3 pos_cc_next;
	float radius;
} vs_out;

void main() {
	vec4 pos_wc = instanceMatrix * vec4(aPos, 1.0);
	vs_out.pos_wc = vec3(pos_wc);
	vec4 pos_cc = view * pos_wc;
	vs_out.pos_cc = pos_cc.xyz / pos_cc.w;

	vec4 pos_wc_next = instanceMatrixNext * vec4(aPos_next, 1.0);
	vec4 pos_cc_next = viewNext * pos_wc_next;
	vs_out.pos_cc_next = pos_cc_next.xyz / pos_cc_next.w;

	gl_Position = projection * pos_cc;
	vs_out.radius = radius;
}

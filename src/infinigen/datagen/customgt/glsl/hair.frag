// FRAGMENT SHADER

#version 440 core

in vec3 interp_pos_cc;
in vec3 interp_pos_cc_next;
in vec3 pa;
in vec3 pb;
in float hair_width;

uniform mat4 wc2img;

layout (location = 0) out ivec4 rasterized_occ_bounds;
layout (location = 1) out vec4 rasterized_cc;
layout (location = 2) out vec4 next_rasterized_cc;
layout (location = 7) out vec4 geo_normal;

float dist_to_line(vec3 a, vec3 b, vec3 p){
    vec3 n = (b - a);
    float numer = length(cross(p-a, n));
    return numer / length(n);
}

vec3 meeting_point(vec3 a, vec3 b, vec3 p){
    vec3 n = (b - a);
    vec3 perp = cross(p-a, n);
	vec3 dir = cross(perp, n);
	dir = normalize(dir) * dist_to_line(a, b, p);
    return p + dir;
}

vec2 proj(vec3 inp){
    vec4 tmp = vec4(inp, 1.0);
    vec4 tmp2 = wc2img * tmp;
    return vec2(tmp2[0]/tmp2[2], tmp2[1]/tmp2[2]);
}

void main() {

	float distance_to_line = dist_to_line(pa, pb, interp_pos_cc);
	if (distance_to_line > hair_width){
		discard;
	}

	float d_normalized = distance_to_line / hair_width;
	float theta = acos(d_normalized);
	float z_delta = sin(theta) * hair_width;

	vec3 vloc_3d = interp_pos_cc + vec3(0., 0., z_delta); // opengl format
	rasterized_cc = vec4(vloc_3d.x, -vloc_3d.y, -vloc_3d.z, 1.0);
	vec3 vloc_3dn = interp_pos_cc_next + vec3(0., 0., z_delta); // opengl format
	next_rasterized_cc = vec4(vloc_3dn.x, -vloc_3dn.y, -vloc_3dn.z, 1.0);

	float approx_hair_width_in_px = length(proj(interp_pos_cc) -  proj(interp_pos_cc + vec3(hair_width, 0.0, 0.0)));
	int occ = int((d_normalized > 0.3) || (approx_hair_width_in_px < 1.1));
	rasterized_occ_bounds = ivec4(0, occ, 0, 1);

	// Geo normal
	vec3 norml = vloc_3d - meeting_point(pa, pb, vloc_3d);
	geo_normal = vec4(norml.x, norml.y, norml.z, 1.0);
}

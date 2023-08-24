// FRAGMENT SHADER

#version 440 core

uniform vec3 cameraPos;

uniform int object_index;

in vec3 interp_pos_wc;
in vec3 interp_pos_cc;
in vec3 interp_pos_cc_next;
in vec3 normal;
in vec3 cc_normal;
in float tri_area;
in float px_area;
flat in float has_flow;
flat in ivec3 instance_id;
flat in ivec3 face_id;
flat in ivec3 tag;

layout (location = 0) out ivec4 rasterized_occ_bounds;
layout (location = 1) out vec4 rasterized_cc;
layout (location = 2) out vec4 next_rasterized_cc;
layout (location = 3) out vec4 rasterized_face_id;
layout (location = 4) out ivec4 object_segmentation;
layout (location = 5) out ivec4 tag_segmentation;
layout (location = 6) out ivec4 instance_segmentation;
layout (location = 7) out vec4 geo_normal;

/*
// No longer used
layout (location = 3) out vec4 faceSize;
layout (location = 4) out vec4 pixelSize;
*/

void main() {
	vec3 updated_normal = normal;
	if (dot((cameraPos - interp_pos_wc), updated_normal) < 0){
		updated_normal = updated_normal * -1;
	}

	rasterized_cc = vec4(interp_pos_cc, 1.0);
	if (has_flow > 0.99)
		next_rasterized_cc = vec4(interp_pos_cc_next, 1.0);
	else
		next_rasterized_cc = vec4(0.0, 0.0, -1.0, 1.0);
	tag_segmentation = ivec4(tag, 0);
	instance_segmentation = ivec4(instance_id[0], instance_id[1], instance_id[2], 1);
	object_segmentation = ivec4(object_index, 0, 0, 1);

	geo_normal = vec4(normalize(-cc_normal), 1.0);
	rasterized_occ_bounds = ivec4(0, 0, 0, 1);

	rasterized_face_id = vec4(face_id, 1.0);

	/*
	// No longer used
	faceSize = vec4(tri_area, 0.0, 0.0, 1.0);
	pixelSize = vec4(px_area, 0.0, 0.0, 1.0);
	*/

}

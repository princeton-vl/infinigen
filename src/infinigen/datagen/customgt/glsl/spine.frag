// FRAGMENT SHADER

#version 440 core

layout (location = 0) out ivec4 rasterized_occ_bounds;

void main() {
	rasterized_occ_bounds = ivec4(0, 1, 0, 1);
}

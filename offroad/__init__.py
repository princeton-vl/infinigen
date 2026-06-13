# Off-road scene generation (path-first) built on top of Infinigen.
#
# Design principle: the road path is AUTHORITATIVE. Terrain and vegetation are
# generated to CONFORM to the road ("environment follows road"), not the other
# way around. Terrain is *anchored* to the road elevation so that no spurious
# plateaus or canyons appear next to it.
#
# This package deliberately does NOT use Infinigen's C/CUDA terrain SDF pipeline
# (which generates terrain independently of any road). It builds its own
# road-anchored ground mesh and then uses Infinigen purely as an asset/scatter/
# material/lighting/render library.

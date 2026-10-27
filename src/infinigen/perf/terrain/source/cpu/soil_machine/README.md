# Optimized SoilMachine erosion

The files here are the ones the optimized erosion changes; everything else is
included from `infinigen/terrain/source/cpu/soil_machine`, which comes after
this directory on the include path. `scripts/install/compile_terrain.sh`
builds them into `terrain/lib/cpu/soil_machine/SoilMachine_perf.so`, which the
optimized terrain loads (`infinigen/perf/terrain/land_process/erosion.py`).
`SoilMachine.cpp` is an unchanged copy, present so that its quoted includes
find these headers first.

The changes compute the same results as the default sources: a constant
instead of a string-keyed lookup, a vector instead of a deque, inlined
accessors, and erosion neighbours sorted by precomputed height keys
(2048x2048, 500k particles: identical SHA256 for both passes).

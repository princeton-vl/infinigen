# Native field operation provenance

`field_ops.hpp` adapts scalar formulas from Blender tag **v4.2.0**:

- `intern/cycles/util/hash.h`: Blender Foundation, Apache-2.0; Jenkins lookup3 hash.
- `intern/cycles/kernel/svm/voronoi.h`: Blender Foundation, Apache-2.0. Distance-to-edge algorithm credits Inigo Quilez (MIT) as in the original source.
- `intern/cycles/kernel/svm/noise.h`: Sony Pictures Imageworks et al. and Blender Foundation, BSD-3-Clause, adapted from Open Shading Language.
- `intern/cycles/kernel/svm/fractal_noise.h`, `map_range.h`, `ramp.h` and `util/math.h`: Blender Foundation, Apache-2.0.
- `extended_ops.hpp` also adapts 1D noise, noise-color offsets from
  `noisetex.h`, and the X-axis sine-band formula from `wave.h`
  (Blender Foundation, Apache-2.0; noise retains its BSD attribution above).

Upstream repository: https://github.com/blender/blender/tree/v4.2.0/intern/cycles

Changes: scalar standalone types, no kernel/global/stack dependencies, distance-only Voronoi outputs, reused candidate vectors in distance-to-edge, explicit direct broadleaf composition in the asset directory. Supported texture configuration is deliberately limited to broadleaf's 3D normalized fBM and 1D/3D Euclidean F1 / distance-to-edge; this is not an implementation of every Blender texture mode.

These adapt Blender's permissively licensed Cycles sources, not its GPL parts, so they are here rather than in the GPL `infinigen_gpl` submodule. Preserve this notice and the license texts (`LICENSE-Apache-2.0.txt`, `LICENSE-BSD-3-Clause.txt`, `LICENSE-MIT.txt`) when redistributing. The asset formulas derive from the repository's BSD-3-Clause leaf sources. The kernels that use these operations are under the same notice: the leaf fields (`infinigen/perf/assets/objects/leaves/native`), the tree bark field (`infinigen/perf/assets/materials/plant/native`) and the pinecone radial noise (`infinigen/perf/assets/objects/monocot/_pinecone/native`).

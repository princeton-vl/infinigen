# Pinecone kernel provenance

`decorate.cpp` and `decorate_avx2.cpp` evaluate the pinecone radial noise with
the native field operations in `infinigen/perf/assets/utils/native`,
which adapt Blender 4.2 Cycles (Apache-2.0) and its OSL-derived Perlin noise
(BSD-3-Clause). Their provenance, attribution and license texts are in that
directory's `NOTICE.md`.

The STUCCI and Simple Deform kernels are GPL-2.0-or-later and come from the
optional `infinigen_gpl` submodule (`infinigen_gpl/perf/pinecone`), where they
are also built.

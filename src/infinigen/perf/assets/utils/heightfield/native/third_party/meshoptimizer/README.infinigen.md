meshoptimizer simplifier
========================

This directory contains `meshoptimizer.h` and `simplifier.cpp` from
meshoptimizer v1.2, pinned at commit
`9d9890c73011d75920af614485296d1e03e95448`. The upstream project is
Copyright (c) 2016-2026 Arseny Kapoulkine and is distributed under the MIT
license reproduced in `LICENSE.md`.

Infinigen uses the geometry-only collapse metric. This copy adds
`meshopt_simplifyWithUpdatePassive`, which transports anonymous numeric fields
after each geometry-selected collapse by sampling the closest local surface at
the accepted quadric target, with bounded edge interpolation at seams.
Passive fields never participate in collapse ranking, geometric error, vertex
positioning, or triangle allocation.

The geometry-only update path ranks integrated quadric error at candidate
replacement points, checks both endpoint neighborhoods for flips, and moves
geometry in the same step as field transfer. A final unpaired position solve
is omitted for this path. Error limits and reported errors retain normalized
distance units independently of the ranking priority. Quadric accumulation
and solving use double precision; returned mesh coordinates remain float32.

Infinigen's separate reconstruction module fits bounded residual corrections
against the original dense raster after simplification. This never changes
topology or vertex positions and has no knowledge of asset or field names.

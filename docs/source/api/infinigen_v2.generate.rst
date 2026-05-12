infinigen_v2.generate
==========================

infinigen_v2.generate is the toplevel CLI entrypoint for most usecases of infinigen_v2

It can execute any generator and any exporter in the system:
- generators include objects, materials, scenes, etc.
- exporters includes rendering images, saving 3D files, computer vision ground truth, etc.

.. argparse::
   :module: infinigen_v2.generate
   :func: get_parser
   :prog: infinigen_v2.generate
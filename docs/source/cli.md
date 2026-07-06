# Command Line Interface

This page documents the command-line entry points shipped with Infinigen.

## `infinigen2`

The main Infinigen 2.0 generation CLI: build one or more generators into a scene and render or export the result.

```{eval-rst}
.. argparse::
   :module: infinigen2.generate
   :func: get_parser
   :prog: infinigen2
```

To list the available generators, run `infinigen2 --list` (optionally filtering by category, e.g. `infinigen2 --list Scene`).

## `infinigen.datagen.manage_jobs`

The Infinigen 1.0 datagen job manager: orchestrate large-scale scene generation across a local machine or a SLURM cluster.

```{eval-rst}
.. argparse::
   :module: infinigen.datagen.manage_jobs
   :func: get_parser
   :prog: python -m infinigen.datagen.manage_jobs
```

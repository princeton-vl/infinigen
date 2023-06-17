#!/usr/bin/env python

# Copyright: 2009-2022 the scikit-image team
# License: BSD-3-Clause
# adapted by Zeyu Ma on date June 5, 2023 to avoid precision loss for large grids to achieve local determinism,
# therefore to make seamless stiched mesh

from skimage._build import cython

import os
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('marching_cubes', parent_package, top_path)

    cython(['marching_cubes/_marching_cubes_lewiner_cy.pyx'], working_path=base_path)

    config.add_extension('_marching_cubes_lewiner_cy',
                         sources=['marching_cubes/_marching_cubes_lewiner_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Graph-based Image-processing Algorithms',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
#!/usr/bin/env python

import os
from setuptools import setup

import lxml
import numpy as np
from Cython.Distutils import Extension
from Cython.Distutils import build_ext


def find_libxml2_include():
    include_dirs = []
    for d in ['/usr/include/libxml2', '/usr/local/include/libxml2']:
        if os.path.exists(os.path.join(d, 'libxml/tree.h')):
            include_dirs.append(d)
    return include_dirs


setup(
    ext_modules=[
        Extension('src.dataset_conversion.blocks',
                  sources=["src/dataset_conversion/blocks.pyx"],
                  include_dirs=lxml.get_include() + find_libxml2_include(),
                  language="c++",
                  libraries=['xml2']),
        Extension('src.dataset_conversion.lcs',
                  sources=["src/dataset_conversion/lcs.pyx"],
                  include_dirs=[np.get_include()],
                  language="c++"),
    ],
    cmdclass={'build_ext': build_ext}
)

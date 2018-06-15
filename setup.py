#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages, Extension

try:
    import lxml
    import numpy as np
    from Cython.Build import cythonize

    # set the flag for building cython
    has_cython = True
except ImportError:
    has_cython = False


def find_libxml2_include():
    """Find lxml2 libraries to include in the extension"""
    include_dirs = []
    for d in ['/usr/include/libxml2', '/usr/local/include/libxml2']:
        if os.path.exists(os.path.join(d, 'libxml/tree.h')):
            include_dirs.append(d)
    return include_dirs


# gracefully handle missing dependencies
def get_ext_modules():
    if has_cython:
        return cythonize([
            Extension('learnhtml.dataset_conversion.blocks', sources=["learnhtml/dataset_conversion/blocks.pyx"],
                      include_dirs=lxml.get_include() + find_libxml2_include(), language="c++", libraries=['xml2']),
            Extension('learnhtml.dataset_conversion.lcs', sources=["learnhtml/dataset_conversion/lcs.pyx"],
                      include_dirs=[np.get_include()], language="c++")
        ])
    return []


def get_cmdclass():
    """Conditionally add build_ext"""
    if has_cython:
        from Cython.Distutils import build_ext
        return {'build_ext': build_ext}
    return {}


PROJECT_NAME = 'LearnHTML'
PROJECT_PACKAGE_NAME = 'learnhtml'
PROJECT_LICENSE = 'Apache License 2.0'
PROJECT_AUTHOR = 'Nichita Uțiu <nikita.utiu@gmail.com>'
PROJECT_COPYRIGHT = ' 2017-2018, {}'.format(PROJECT_AUTHOR)

PROJECT_DESCRIPTION = 'Machine learning library for content extraction'
# generate the github link for download
PROJECT_GITHUB_USERNAME = 'nikitautiu'

PROJECT_GITHUB_REPOSITORY = 'learnhtml'
GITHUB_PATH = '{}/{}'.format(
    PROJECT_GITHUB_USERNAME, PROJECT_GITHUB_REPOSITORY)

GITHUB_URL = 'https://github.com/{}'.format(GITHUB_PATH)

DOWNLOAD_URL = '{}/archive/{}.zip'.format(GITHUB_URL, 'master')
# get the packages
PACKAGES = find_packages(exclude=['tests', 'tests.*'])
PACKAGE_DATA = {'': ['*.pyx', '*.pxd', '*.c', '*.h'],
                'learnhtml.cli': ['prepare_data.sh'],
                'learnhrml': ['data/*']}

EXT_MODULES = get_ext_modules()
# requirements
REQUIRES = [
    'tensorflow==1.8.0',
    'click==6.7',
    'click_log==0.3.2',
    'dask[complete]==0.17.5',
    'keras==2.2.0',
    'pandas>=0.23.1',
    'scipy==1.1.0',
    'scikit_learn==0.19.1',
    'sparse==0.3.1',
    'lxml==4.2.1',
    'numpy>=1.14.3'
]

SETUP_REQUIRES = [
    'Cython',
    'lxml==4.2.1',
    'numpy>=1.14.3'
]

setup(
    name=PROJECT_PACKAGE_NAME,
    version='0.1',
    license=PROJECT_LICENSE,
    download_url=DOWNLOAD_URL,
    author=PROJECT_AUTHOR,
    description=PROJECT_DESCRIPTION,
    url=GITHUB_URL,
    packages=PACKAGES,
    ext_modules=EXT_MODULES,
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    setup_requires=SETUP_REQUIRES,
    install_requires=REQUIRES,
    python_requires='>={}'.format(3.5),
    test_suite='tests',
    keywords=['scraping', 'machine learning', 'web content extraction'],
    cmdclass=get_cmdclass(),
    entry_points={
        'console_scripts': [
            'learnhtml = learnhtml.cli.script:script'
        ],
    },
)

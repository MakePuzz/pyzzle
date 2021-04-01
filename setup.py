#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from setuptools import find_packages
import imp
with open("README.md") as f:
    readme = f.read()

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace('"', '').replace("'", '')
                for line in open(os.path.join(here, 'pyzzle', '__init__.py'))
                if line.startswith('__version__ = ')), '0.0.dev0')

setup_params = {
    "name": "pyzzle",
    "version": version,
    "url": "https://github.com/MakePuzz/pyzzle",
    "author": "The MakePuzz team",
    "author_email": "puzzle.hokkaido@gmail.com",
    "maintainer": "The MakePuzz team",
    "maintainer_email": "puzzle.hokkaido@gmail.com",
    "description": "A Python library to automatically generate intelligent and beautiful puzzles",
    "long_description": readme,
    "packages": find_packages(),
    "install_requires": open("requirements.txt").read().splitlines(),
    "license": "GPLv3+",
    "python_requires": ">=3.6",
    "classifiers": [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: GNU General Public License v3 " +
        "or later (GPLv3+)",
    ],
}

try:
    imp.find_module('numpy')
    extensions = []
    from numpy.distutils.core import setup, Extension

    extensions.append(
        Extension(name="pyzzle.Puzzle.add_to_limit",
                  sources=["pyzzle/Puzzle/add_to_limit.f90"]))  # f2py_options=["--opt='-O3'"]
    setup_params.update({"ext_modules": extensions})
    setup(**setup_params)

except ImportError:
    from distutils.core import setup
    setup(**setup_params)

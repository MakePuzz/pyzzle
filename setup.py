#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from setuptools import find_packages

with open("README.md") as f:
    readme = f.read()

extensions = []
from numpy.distutils.core import setup, Extension
extensions.append(
    Extension(name="pyzzle.Puzzle.add_to_limit",
              sources=["pyzzle/Puzzle/add_to_limit.f90"])) #f2py_options=["--opt='-O3'"]

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace('"', '').replace("'", '')
                for line in open(os.path.join(here, 'pyzzle', '__init__.py'))
                if line.startswith('__version__ = ')), '0.0.dev0')

setup(
    name="pyzzle",
    version=version,
    url="https://github.com/MakePuzz/pyzzle",
    author="The MakePuzz team",
    author_email="puzzle.hokkaido@gmail.com",
    maintainer="The MakePuzz team",
    maintainer_email="puzzle.hokkaido@gmail.com",
    description="A Python library to automatically generate intelligent and beautiful puzzles",
    long_description=readme,
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=open("requirements.txt").read().splitlines(),
    license="GPLv3+",
    python_requires=">=3.6",
    classifiers=[
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
)
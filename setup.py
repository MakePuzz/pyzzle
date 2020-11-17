#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension



try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''

def _requires_from_file(filename):
    return open(filename).read().splitlines()

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace("'", '')
                for line in open(os.path.join(here,
                                              'pyzzle',
                                              '__init__.py'))
                if line.startswith('__version__ = ')),
               '0.0.dev0')

extensions = []
extensions.append(
    Extension(name='pyzzle.Puzzle.add_to_limit',
              sources=['pyzzle/Puzzle/add_to_limit.f90'])) #f2py_options=["--opt='-O3'"]

setup(
    name="pyzzle",
    version=version,
    url='https://github.com/puzzle-japan/pyzzle',
    author='The puzzle-japan Team',
    author_email='puzzle.hokkaido@gmail.com',
    maintainer='tsukada-cs and Saikoro2007',
    maintainer_email='puzzle.hokkaido@gmail.com',
    description='A Python library to automatically generate intelligent and beautifull puzzles',
    long_description=readme,
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=_requires_from_file('requirements.txt'),
    license="MIT",
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points="""
      # -*- Entry points: -*-
      [console_scripts]
      pyzzle = pyzzle.script.command:main
    """,
)
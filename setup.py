#!/usr/bin/env python

from distutils.core import setup

setup(
  name = 'audspec',
  version='1.0.0',
  py_modules = ['audspec'],
  classifiers = [
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Multimedia :: Sound/Audio :: Speech'
  ],
  requires = [
    'librosa',
    'numpy',
    'scipy',
]

)

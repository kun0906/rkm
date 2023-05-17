"""
    Compile the CPP code with the command:
        python setup.py build

    https://nesi.github.io/perf-training/python-scatter/ctypes

    https://stackoverflow.com/questions/49000674/cython-std-c11-error-using-both-c-and-c
"""

from setuptools import setup, Extension
# Importing sys module
import sys
path = sys.path
# Getting the default path of the Python interpreter
print(path)
# lib_dirs = '/Users/kunyang/opt/miniconda3/envs/rkm/'

# Compile *TukeyDepth.cpp* into a shared library
setup(
    #...
    ext_modules=[Extension('TukeyDepth', ['TukeyDepth.cpp'],
                           language="c++", extra_compile_args=["-std=c++11"],
                           #  include_dirs=[lib_dirs + '/include'],
                           # library_dirs=[lib_dirs + '/lib'],
                           libraries=['armadillo'],     # Third-party libraries
                           ),],

)


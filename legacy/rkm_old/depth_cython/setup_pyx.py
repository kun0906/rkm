"""

python setup.py build

"""
from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='Depth',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("TukeyDepth2",
                 sources=["TukeyDepth.pyx"],
                 language="c++", extra_compile_args=["-std=c++11"],
                 libraries=['armadillo'],     # Third-party libraries
                 include_dirs=[numpy.get_include()])],
    author='Kun Yang',
    author_email='kun.bj@outlook.com'

)
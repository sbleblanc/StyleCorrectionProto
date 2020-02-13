from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("stylecorrection/utils/cython_utils.pyx")
)
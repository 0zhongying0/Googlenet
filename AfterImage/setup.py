from distutils.core import setup
# from Cython.Build import cythonize
import Cython
from Cython import Build
setup(
    ext_modules = Cython.Build.cythonize(["AfterImage/*.pyx"])
)
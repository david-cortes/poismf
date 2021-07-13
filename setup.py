try:
    import setuptools
    from setuptools import setup
    from setuptools import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy
from sys import platform
import os
from Cython.Distutils import build_ext
from sys import platform
import sys, os


## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio is 'special'
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c99']
                e.extra_link_args += ['-fopenmp']
                # e.extra_compile_args += ["-fsanitize=address", "-static-libasan", "-ggdb"]
                # e.extra_link_args += ["-fsanitize=address", "-static-libasan"]
                # e.extra_compile_args += ["-ggdb"]

        ## Note: apple will by default alias 'gcc' to 'clang', and will ship its own "special"
        ## 'clang' which has no OMP support and nowadays will purposefully fail to compile when passed
        ## '-fopenmp' flags. If you are using mac, and have an OMP-capable compiler,
        ## comment out the code below, or set 'use_omp' to 'True'.
        if not use_omp:
            for e in self.extensions:
                e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
                e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']

        build_ext.build_extensions(self)

use_omp = (("enable-omp" in sys.argv)
           or ("-enable-omp" in sys.argv)
           or ("--enable-omp" in sys.argv))
if use_omp:
    sys.argv = [a for a in sys.argv if a not in ("enable-omp", "-enable-omp", "--enable-omp")]
if os.environ.get('ENABLE_OMP') is not None:
    use_omp = True
if platform[:3] != "dar":
    use_omp = True

### Shorthand for apple computer:
### uncomment line below
# use_omp = True


from_rtd = os.environ.get('READTHEDOCS') == 'True'
if not from_rtd:
    setup(
        name  = "poismf",
        packages = ["poismf"],
        author = 'David Cortes',
        author_email = 'david.cortes.rivera@gmail.com',
        url = 'https://github.com/david-cortes/poismf',
        version = '0.3.1',
        install_requires = ['numpy', 'pandas>=0.24', 'cython', 'scipy'],
        description = 'Fast and memory-efficient Poisson factorization for sparse count matrices',
        cmdclass = {'build_ext': build_ext_subclass},
        ext_modules = [
            Extension("poismf.c_funs_double",
                sources=["poismf/cfuns_double.pyx",
                         "src/poismf.c", "src/topN.c", "src/pred.c",
                         "src/nonnegcg.c", "src/tnc.c"],
                include_dirs=[numpy.get_include(), "src/"],
                define_macros = [("_FOR_PYTHON", None)]),
            Extension("poismf.c_funs_float",
                sources=["poismf/cfuns_float.pyx",
                         "src/poismf.c", "src/topN.c", "src/pred.c",
                         "src/nonnegcg.c", "src/tnc.c"],
                include_dirs=[numpy.get_include(), "src/"],
                define_macros = [("_FOR_PYTHON", None), ("USE_FLOAT", None)])
            ]
    )

    if not use_omp:
        import warnings
        apple_msg  = "\n\n\nMacOS detected. Package will be built without multi-threading capabilities, "
        apple_msg += "due to Apple's lack of OpenMP support in default clang installs. In order to enable it, "
        apple_msg += "install the package directly from GitHub: https://www.github.com/david-cortes/poismf\n"
        apple_msg += "Using 'python setup.py install enable-omp'. "
        apple_msg += "You'll also need an OpenMP-capable compiler.\n\n\n"
        warnings.warn(apple_msg)
else:
    setup(
        name  = "poismf",
        packages = ["poismf"],
        author = 'David Cortes',
        author_email = 'david.cortes.rivera@gmail.com',
        url = 'https://github.com/david-cortes/poismf',
        version = '0.3.1',
        install_requires = ['numpy', 'scipy', 'pandas>=0.24', 'cython'],
        description = 'Fast and memory-efficient Poisson factorization for sparse count matrices',
    )

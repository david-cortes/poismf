try:
    from setuptools import setup
    from setuptools import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy
from sys import platform
import os
from findblas.distutils import build_ext_with_blas
from sys import platform


## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio is 'special'
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99']
                e.extra_link_args += ['-fopenmp']
        build_ext_with_blas.build_extensions(self)


setup(
    name  = "poismf",
    packages = ["poismf"],
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/poismf',
    version = '0.1.0',
    install_requires = ['numpy', 'pandas', 'cython', 'findblas'],
    description = 'Fast and memory-efficient Poisson factorization for sparse count matrices',
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension("poismf.poismf_c_wrapper", sources=["poismf/poismf_c_wrapper.pyx", "src/nonnegcg.c"],
        include_dirs=[numpy.get_include()], define_macros = [("_FOR_PYTHON", None)]
        )]
    )

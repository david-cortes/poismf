from distutils.core import setup
from distutils.extension import Extension
import numpy
from sys import platform
import os
from findblas.distutils import build_ext_with_blas


## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2']
        elif compiler == 'ming32':
            # mingw32 doesn't support OpenMP in a default conda install
            # you can enable it by putting in 'extra_compile_args'
            # the following entry '-fopenmp=libomp5 <path_to_libomp.so or .a>'
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-march=native', '-std=c99']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99']
                e.extra_link_args += ['-fopenmp']
        build_ext_with_blas.build_extensions(self)


setup(
    name  = "poismf",
    packages = ["poismf"],
    author = 'David Cortes',
    url = 'https://github.com/david-cortes/poismf',
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension("poismf", sources=["poismf/pmf.pyx"], include_dirs=[numpy.get_include()])]
    )

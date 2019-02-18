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
        if compiler == 'msvc': # visual studio is 'special'
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99']
                e.extra_link_args += ['-fopenmp']
        build_ext_with_blas.build_extensions(self)

### Get include and link path of package nonnegcg
import nonnegcg, re, platform as ptf, sys
nncg_lib_dir = re.sub(r"__init__\.py$", "", nonnegcg.__file__)
if ptf.system() == "Windows":
    lib_ext_regex = r"\.pyd$"
else:
    lib_ext_regex = r"\.so$"
nncg_lib_file = [f for f in os.listdir(nncg_lib_dir) if bool(re.search(lib_ext_regex, f))]
if len(nncg_lib_file) == 0:
    raise ValueError("Must install dependency 'nonnegcg' with shared object (https://www.github.com/david-cortes/nonneg_cg)")
nncg_lib_file = nncg_lib_file[0]
nncg_inc_file = [f for f in os.listdir(nncg_lib_dir) if bool(re.search(r"\.h", f))]
if len(nncg_inc_file) > 0:
    nncg_inc_path = nncg_lib_dir
else:
    if os.path.exists(os.path.join(sys.prefix, "include", "nonnegcg.h")):
        nncg_inc_path = os.path.join(sys.prefix, "include")
    else:
        raise ValueError("Could not find header file for 'nonnegcg' (https://www.github.com/david-cortes/nonneg_cg)")
if ptf.system() == "Windows":
    link_args = [os.path.join(nncg_lib_dir, nncg_lib_file)]
else:
    link_args = ["-L" + nncg_lib_dir, "-l:" + nncg_lib_file]


setup(
    name  = "poismf",
    packages = ["poismf"],
    author = 'David Cortes',
    url = 'https://github.com/david-cortes/poismf',
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension("poismf", sources=["poismf/pmf.pyx"], extra_link_args = link_args,
        include_dirs=[numpy.get_include(), nncg_inc_path, nncg_lib_dir], define_macros = [("_FOR_PYTHON", None)],
        runtime_library_dirs = [nncg_lib_dir]
        )]
    )

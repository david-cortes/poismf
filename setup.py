try:
    import setuptools
    from setuptools import setup
    from setuptools import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy
import sys, os, subprocess, warnings, re
from Cython.Distutils import build_ext

found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        if self.compiler.compiler_type == 'msvc':
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            self.add_march_native()
            self.add_openmp_linkage()

            for e in self.extensions:
                # e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c99']
                # e.extra_link_args += ['-fopenmp']
                
                # e.extra_compile_args += ["-fsanitize=address", "-static-libasan", "-ggdb"]
                # e.extra_link_args += ["-fsanitize=address", "-static-libasan"]
                # e.extra_compile_args += ["-ggdb"]
                
                e.extra_compile_args += ['-O3', '-std=c99']

        build_ext.build_extensions(self)

    def add_march_native(self):
        arg_march_native = "-march=native"
        arg_mcpu_native = "-mcpu=native"
        if self.test_supports_compile_arg(arg_march_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_march_native)
        elif self.test_supports_compile_arg(arg_mcpu_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_mcpu_native)

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        if self.test_supports_compile_arg(arg_omp1):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif self.test_supports_compile_arg(arg_omp2):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        else:
            set_omp_false()

    def test_supports_compile_arg(self, comm):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "poismf_compiler_testing.c"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                cmd = [self.compiler.compiler[0]]
            except:
                cmd = list(self.compiler.compiler)
            val_good = subprocess.call(cmd + [fname])
            try:
                val = subprocess.call(cmd + comm + [fname])
                is_supported = (val == val_good)
            except:
                is_supported = False
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        return is_supported


from_rtd = os.environ.get('READTHEDOCS') == 'True'
if not from_rtd:
    setup(
        name  = "poismf",
        packages = ["poismf"],
        author = 'David Cortes',
        author_email = 'david.cortes.rivera@gmail.com',
        url = 'https://github.com/david-cortes/poismf',
        version = '0.3.1-4',
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

    if not found_omp:
        omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
        omp_msg += " To enable multi-threading, first install OpenMP"
        if (sys.platform[:3] == "dar"):
            omp_msg += " - for macOS: 'brew install libomp'\n"
        else:
            omp_msg += " modules for your compiler. "
        
        omp_msg += "Then reinstall this package from scratch: 'pip install --force-reinstall poismf'.\n"
        warnings.warn(omp_msg)
else:
    setup(
        name  = "poismf",
        packages = ["poismf"],
        author = 'David Cortes',
        author_email = 'david.cortes.rivera@gmail.com',
        url = 'https://github.com/david-cortes/poismf',
        version = '0.3.1-4',
        install_requires = ['numpy', 'scipy', 'pandas>=0.24', 'cython'],
        description = 'Fast and memory-efficient Poisson factorization for sparse count matrices',
    )

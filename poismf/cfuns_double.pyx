#cython: freethreading_compatible=True, language_level=2
import ctypes

ctypedef double real_t
c_real = ctypes.c_double
include "poismf_c_wrapper.pxi"

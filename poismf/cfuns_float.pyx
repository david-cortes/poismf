#cython: freethreading_compatible=True, language_level=2
import ctypes

ctypedef float real_t
c_real = ctypes.c_float
include "poismf_c_wrapper.pxi"

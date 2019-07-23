import numpy as np
cimport numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sys import platform

cdef extern from "../src/pgd.c":
	void run_poismf(
		double *A, double *Xr, size_t *Xr_indptr, size_t *Xr_indices,
		double *B, double *Xc, size_t *Xc_indptr, size_t *Xc_indices,
		size_t dimA, size_t dimB, size_t k,
		double l2_reg, double l1_reg, int use_cg, double step_size,
		size_t numiter, size_t npass, int ncores)

def run_pgd(Xcoo, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, int use_cg=0, double l2_reg=1e9, double l1_reg=0, double step_size=1e-7, size_t niter=10, size_t npass=1, int ncores=1):

	if use_cg and (platform[:3] == "win"):
		raise ValueError("CG method not available in Windows OS.")

	Xcsr = csr_matrix(Xcoo)
	Xcsc = csc_matrix(Xcoo)

	cdef np.ndarray[double, ndim=1] Xr = Xcsr.data.astype(ctypes.c_double)
	cdef np.ndarray[size_t, ndim=1] Xr_indices = Xcsr.indices.astype(ctypes.c_size_t)
	cdef np.ndarray[size_t, ndim=1] Xr_indptr = Xcsr.indptr.astype(ctypes.c_size_t)
	cdef size_t dimA = A.shape[0]

	cdef np.ndarray[double, ndim=1] Xc = Xcsc.data.astype(ctypes.c_double)
	cdef np.ndarray[size_t, ndim=1] Xc_indices = Xcsc.indices.astype(ctypes.c_size_t)
	cdef np.ndarray[size_t, ndim=1] Xc_indptr = Xcsc.indptr.astype(ctypes.c_size_t)
	cdef size_t dimB = B.shape[0]

	cdef size_t k = A.shape[1]

	run_poismf(
		&A[0,0], &Xr[0], &Xr_indptr[0], &Xr_indices[0],
		&B[0,0], &Xc[0], &Xc_indptr[0], &Xc_indices[0],
		dimA, dimB, k,
		l2_reg, l1_reg, use_cg, step_size,
		niter, npass, ncores
		)

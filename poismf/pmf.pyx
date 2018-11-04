import numpy as np
cimport numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

cdef extern from "pgd.c":
	void optimize(double *A, double *Xr, size_t *Xr_indices, size_t *Xr_indptr,
		double *B, double* Xc, size_t *Xc_indices, size_t *Xc_indptr,
		size_t dimA, size_t dimB, size_t nnz, size_t k,
		double reg_param, double step_size,
		size_t maxiter, size_t npass, int ncores
		)

def run_pgd(Xcoo, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, double reg_param=1e7, double step_size=1e-5, size_t niter=10, size_t npass=1, int ncores=1):
	Xcsr = csr_matrix(Xcoo)
	Xcsc = csc_matrix(Xcoo)

	cdef np.ndarray[double, ndim=1] Xr = Xcsr.data.astype('float64')
	cdef np.ndarray[size_t, ndim=1] Xr_indices = Xcsr.indices.astype(ctypes.c_size_t)
	cdef np.ndarray[size_t, ndim=1] Xr_indptr = Xcsr.indptr.astype(ctypes.c_size_t)
	cdef size_t dimA = Xcoo.shape[0]

	cdef np.ndarray[double, ndim=1] Xc = Xcsc.data.astype('float64')
	cdef np.ndarray[size_t, ndim=1] Xc_indices = Xcsc.indices.astype(ctypes.c_size_t)
	cdef np.ndarray[size_t, ndim=1] Xc_indptr = Xcsc.indptr.astype(ctypes.c_size_t)
	cdef size_t dimB = Xcoo.shape[1]

	cdef size_t nnz = Xcsr.data.shape[0]
	cdef size_t k = A.shape[1]

	optimize(
		&A[0,0], &Xr[0], &Xr_indices[0], &Xr_indptr[0],
		&B[0,0], &Xc[0], &Xc_indices[0], &Xc_indptr[0],
		dimA, dimB, nnz, k,
		reg_param, step_size,
		niter, npass, ncores
		)


import numpy as np
cimport numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

IF UNAME_SYSNAME == "Windows":
	obj_ind_type = ctypes.c_long
	ctypedef long ind_type
ELSE:
	obj_ind_type = ctypes.c_size_t
	ctypedef size_t ind_type

cdef extern from "pgd.c":
	void run_poismf(double *A, double *Xr, ind_type *Xr_indptr, ind_type *Xr_indices,
		double *B, double* Xc, ind_type *Xc_indptr, ind_type *Xc_indices,
		ind_type dimA, ind_type dimB, ind_type nnz, ind_type k,
		double reg_param, double step_size,
		ind_type maxiter, ind_type npass, int ncores
		)

def run_pgd(Xcoo, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, double reg_param=1e9, double step_size=1e-7, ind_type niter=10, ind_type npass=1, int ncores=1):
	Xcsr = csr_matrix(Xcoo)
	Xcsc = csc_matrix(Xcoo)

	cdef np.ndarray[double, ndim=1] Xr = Xcsr.data.astype('float64')
	cdef np.ndarray[ind_type, ndim=1] Xr_indices = Xcsr.indices.astype(obj_ind_type)
	cdef np.ndarray[ind_type, ndim=1] Xr_indptr = Xcsr.indptr.astype(obj_ind_type)
	cdef ind_type dimA = A.shape[0]

	cdef np.ndarray[double, ndim=1] Xc = Xcsc.data.astype('float64')
	cdef np.ndarray[ind_type, ndim=1] Xc_indices = Xcsc.indices.astype(obj_ind_type)
	cdef np.ndarray[ind_type, ndim=1] Xc_indptr = Xcsc.indptr.astype(obj_ind_type)
	cdef ind_type dimB = B.shape[0]

	cdef ind_type nnz = Xcsr.data.shape[0]
	cdef ind_type k = A.shape[1]

	run_poismf(
		&A[0,0], &Xr[0], &Xr_indptr[0], &Xr_indices[0],
		&B[0,0], &Xc[0], &Xc_indptr[0], &Xc_indices[0],
		dimA, dimB, nnz, k,
		reg_param, step_size,
		niter, npass, ncores
		)

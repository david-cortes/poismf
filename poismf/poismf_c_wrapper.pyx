import numpy as np
cimport numpy as np

cdef extern from "../src/pgd.c":
	void run_poismf(
		double *A, double *Xr, size_t *Xr_indptr, size_t *Xr_indices,
		double *B, double *Xc, size_t *Xc_indptr, size_t *Xc_indices,
		size_t dimA, size_t dimB, size_t k,
		double l2_reg, double l1_reg, int use_cg, double step_size,
		size_t numiter, size_t npass, int ncores)
	void optimize_cg_single(double *curr, double *X, size_t *X_ind, size_t nnz_this, double *F, double *Fsum, int k, double l2_reg)
	void predict_multiple(double *out, double *A, double *B, size_t *ix_u, size_t *ix_i, size_t n, int k, int nthreads)

def run_pgd(np.ndarray[double, ndim=1] Xr, np.ndarray[size_t, ndim=1] Xr_indices, np.ndarray[size_t, ndim=1] Xr_indptr,
			np.ndarray[double, ndim=1] Xc, np.ndarray[size_t, ndim=1] Xc_indices, np.ndarray[size_t, ndim=1] Xc_indptr,
			np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B,
			int use_cg=0, double l2_reg=1e9, double l1_reg=0, double step_size=1e-7, size_t niter=10, size_t npass=1, int nthreads=1):

	cdef size_t dimA = A.shape[0]
	cdef size_t dimB = B.shape[0]
	cdef size_t k = A.shape[1]

	run_poismf(
		&A[0,0], &Xr[0], &Xr_indptr[0], &Xr_indices[0],
		&B[0,0], &Xc[0], &Xc_indptr[0], &Xc_indices[0],
		dimA, dimB, k,
		l2_reg, l1_reg, use_cg, step_size,
		niter, npass, nthreads
		)

def _predict_multiple(np.ndarray[double, ndim=1] out, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B,
					  np.ndarray[size_t, ndim=1] ix_u, np.ndarray[size_t, ndim=1] ix_i, int nthreads):
	predict_multiple(&out[0], &A[0,0], &B[0,0], &ix_u[0], &ix_i[0], ix_u.shape[0], A.shape[1], nthreads)

def _predict_factors(np.ndarray[double, ndim=1] a_init, np.ndarray[double, ndim=1] counts, np.ndarray[size_t, ndim=1] ix,
					 np.ndarray[double, ndim=2] B, np.ndarray[double, ndim=1] Bsum, double l2_reg, double l1_reg):
	if l1_reg > 0:
		Bsum += l1_reg
	optimize_cg_single(&a_init[0], &counts[0], &ix[0], counts.shape[0], &B[0,0], &Bsum[0], B.shape[1], l2_reg)

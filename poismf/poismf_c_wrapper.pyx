import numpy as np
cimport numpy as np
import ctypes

cdef extern from "../src/poismf.h":
    ctypedef size_t sparse_ix
    ctypedef enum Method:
        tncg = 1
        cg = 2
        pg = 3
    int run_poismf(
        double *A, double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
        double *B, double *Xc, sparse_ix *Xc_indptr, sparse_ix *Xc_indices,
        const size_t dimA, const size_t dimB, const size_t k,
        const double l2_reg, const double l1_reg, const double w_mult, double step_size,
        const Method method, const bint limit_step, const size_t numiter, const size_t maxupd,
        const int nthreads)
    int factors_single(
        double *out, size_t k,
        double *A_old, size_t dimA,
        double *X, sparse_ix X_ind[], size_t nnz,
        double *B, double *Bsum,
        int maxupd, double l2_reg, double l1_new, double l1_old,
        double w_mult
    )
    void predict_multiple(
        double *out,
        double *A, double *B,
        sparse_ix *ixA, sparse_ix *ixB,
        size_t n, int k,
        int nthreads
    )
    long double eval_llk(
        double *A,
        double *B,
        sparse_ix ixA[],
        sparse_ix ixB[],
        double *X,
        size_t nnz, int k,
        bint full_llk, bint include_missing,
        size_t dimA, size_t dimB,
        int nthreads
    )
    int factors_multiple(
        double *A, double *B, double *A_old, double *Bsum,
        double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
        int k, size_t dimA,
        double l2_reg, double w_mult,
        double step_size, size_t niter, size_t maxupd,
        Method method, bint limit_step,
        int nthreads
    )
    int topN(
        double *a_vec, double *B, int k,
        sparse_ix *include_ix, size_t n_include,
        sparse_ix *exclude_ix, size_t n_exclude,
        sparse_ix *outp_ix, double *outp_score,
        size_t n_top, size_t n, int nthreads
    )

def _run_poismf(
    np.ndarray[double, ndim=1] Xr,
    np.ndarray[size_t, ndim=1] Xr_indices,
    np.ndarray[size_t, ndim=1] Xr_indptr,
    np.ndarray[double, ndim=1] Xc,
    np.ndarray[size_t, ndim=1] Xc_indices,
    np.ndarray[size_t, ndim=1] Xc_indptr,
    np.ndarray[double, ndim=2] A,
    np.ndarray[double, ndim=2] B,
    method="tncg", bint limit_step=0,
    double l2_reg=1e9, double l1_reg=0, double w_mult=1.,
    double step_size=1e-7,
    size_t niter=10, size_t maxupd=1, int nthreads=1):

    cdef size_t dimA = A.shape[0]
    cdef size_t dimB = B.shape[0]
    cdef size_t k = A.shape[1]
    cdef Method c_method = tncg
    if method == "tncg":
        c_method = tncg
    elif method == "cg":
        c_method = cg
    elif method == "pg":
        c_method = pg

    cdef int ret_code = run_poismf(
        &A[0,0], &Xr[0], &Xr_indptr[0], &Xr_indices[0],
        &B[0,0], &Xc[0], &Xc_indptr[0], &Xc_indices[0],
        dimA, dimB, k,
        l2_reg, l1_reg, w_mult, step_size,
        c_method, limit_step, niter, maxupd, nthreads
        )
    if ret_code == 1:
        raise MemoryError("Could not allocate enough memory.")

def _predict_multiple(np.ndarray[double, ndim=1] out, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B,
                      np.ndarray[size_t, ndim=1] ix_u, np.ndarray[size_t, ndim=1] ix_i, int nthreads):
    predict_multiple(&out[0], &A[0,0], &B[0,0], &ix_u[0], &ix_i[0], ix_u.shape[0], A.shape[1], nthreads)

def _predict_factors(
        np.ndarray[double, ndim=2] A_old,
        np.ndarray[double, ndim=1] counts,
        np.ndarray[size_t, ndim=1] ix,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=1] Bsum,
        size_t maxupd = 20,
        double l2_reg = 1e5,
        double l1_new = 0., double l1_old = 0.,
        double w_mult = 1., bint limit_step = 0,
    ):
    cdef np.ndarray[double, ndim=1] out = np.empty(A_old.shape[1], dtype=ctypes.c_double)
    cdef int ret_code = factors_single(
        &out[0], <size_t> A_old.shape[1],
        &A_old[0,0], <size_t> A_old.shape[0],
        &counts[0], &ix[0], <size_t> counts.shape[0],
        &B[0,0], &Bsum[0],
        maxupd, l2_reg, l1_new, l1_old,
        w_mult
    )
    if ret_code != 0:
        raise MemoryError("Could not allocate enough memory.")
    return out

def _predict_factors_multiple(
        np.ndarray[double, ndim=2] A_old,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=1] Bsum,
        np.ndarray[size_t, ndim=1] Xr_indptr,
        np.ndarray[size_t, ndim=1] Xr_indices,
        np.ndarray[double, ndim=1] Xr,
        double l2_reg = 1e9,
        double w_mult = 1.,
        double step_size = 1e-7,
        size_t niter = 10,
        size_t maxupd = 1,
        method = "tncg",
        bint limit_step = 0,
        int nthreads = 1
    ):
    cdef int k = <size_t> B.shape[1]
    cdef size_t dimA = Xr_indptr.shape[0] - 1
    cdef np.ndarray[double, ndim=2] A = np.empty((dimA, k), dtype=ctypes.c_double)

    cdef double *ptr_A = &A[0,0]
    cdef double *ptr_B = &B[0,0]
    cdef double *ptr_A_old = &A_old[0,0]
    cdef double *ptr_Bsum = &Bsum[0]
    cdef size_t *ptr_Xr_indptr = &Xr_indptr[0]
    cdef size_t *ptr_Xr_indices = &Xr_indices[0]
    cdef double *ptr_Xr = &Xr[0]

    cdef Method c_method = tncg
    if method == "tncg":
        c_method = tncg
    elif method == "cg":
        c_method = cg
    elif method == "pg":
        c_method = pg

    cdef int returned_val = factors_multiple(
        ptr_A, ptr_B, ptr_A_old, ptr_Bsum,
        ptr_Xr, ptr_Xr_indptr, ptr_Xr_indices,
        k, dimA,
        l2_reg, w_mult,
        step_size, niter, maxupd,
        c_method, limit_step,
        nthreads
    )

    if returned_val:
        raise MemoryError("Could not allocate enough memory.")

    return A

def _eval_llk_test(
        np.ndarray[double, ndim=2] A,
        np.ndarray[double, ndim=2] B,
        np.ndarray[size_t, ndim=1] ixA,
        np.ndarray[size_t, ndim=1] ixB,
        np.ndarray[double, ndim=1] X,
        bint full_llk = 0, bint include_missing = 0,
        int nthreads = 1
    ):
    cdef double *ptr_A = &A[0,0]
    cdef double *ptr_B = &B[0,0]
    cdef size_t *ptr_ixA = &ixA[0]
    cdef size_t *ptr_ixB = &ixB[0]
    cdef double *ptr_X = &X[0]
    cdef size_t nnz = X.shape[0]
    cdef size_t dimA = A.shape[0]
    cdef size_t dimB = B.shape[0]
    cdef int k = A.shape[1]
    cdef long double res = eval_llk(
        ptr_A,
        ptr_B,
        ptr_ixA,
        ptr_ixB,
        ptr_X,
        nnz, k,
        full_llk, include_missing,
        dimA, dimB,
        nthreads
    )
    if np.isnan(res):
        raise MemoryError("Could not allocate enough memory.")
    return res

def _call_topN(
        np.ndarray[double, ndim=1] a_vec,
        np.ndarray[double, ndim=2] B,
        np.ndarray[size_t, ndim=1] include_ix,
        np.ndarray[size_t, ndim=1] exclude_ix,
        size_t top_n = 10,
        bint output_score = 0,
        int nthreads = 1
    ):
    cdef double *ptr_a = &a_vec[0]
    cdef double *ptr_B = &B[0,0]
    cdef int k = B.shape[1]
    cdef size_t n = B.shape[0]
    cdef size_t *ptr_include = NULL
    cdef size_t *ptr_exclude = NULL
    cdef size_t n_include = 0
    cdef size_t n_exclude = 0
    if include_ix.shape[0]:
        n_include = include_ix.shape[0]
        ptr_include = &include_ix[0]
    elif exclude_ix.shape[0]:
        n_exclude = exclude_ix.shape[0]
        ptr_exclude = &exclude_ix[0]

    cdef np.ndarray[size_t, ndim=1] outp_ix = np.empty(top_n, dtype=ctypes.c_size_t)
    cdef np.ndarray[double, ndim=1] outp_score = np.empty(0, dtype=ctypes.c_double)
    cdef size_t *ptr_outp_ix = &outp_ix[0]
    cdef double *ptr_outp_score = NULL
    if output_score:
        outp_score = np.empty(top_n, dtype=ctypes.c_double)
        ptr_outp_score = &outp_score[0]

    topN(
        ptr_a, ptr_B, k,
        ptr_include, n_include,
        ptr_exclude, n_exclude,
        ptr_outp_ix, ptr_outp_score,
        top_n, n, nthreads
    )
    return outp_ix, outp_score

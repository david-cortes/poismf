import numpy as np
cimport numpy as np
from cython cimport boundscheck, nonecheck, wraparound
import ctypes

cdef extern from "../src/poismf.h":
    ctypedef size_t sparse_ix
    ctypedef enum Method:
        tncg = 1
        cg = 2
        pg = 3
    int run_poismf(
        real_t *A, real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
        real_t *B, real_t *Xc, sparse_ix *Xc_indptr, sparse_ix *Xc_indices,
        const size_t dimA, const size_t dimB, const size_t k,
        const real_t l2_reg, const real_t l1_reg, const real_t w_mult, real_t step_size,
        const Method method, const bint limit_step, const size_t numiter, const size_t maxupd,
        const bint early_stop, const bint reuse_prev,
        const bint handle_interrupt, const int nthreads) nogil
    int factors_single(
        real_t *out, size_t k,
        real_t *Amean, bint reuse_mean,
        real_t *X, sparse_ix X_ind[], size_t nnz,
        real_t *B, real_t *Bsum,
        int maxupd, real_t l2_reg, real_t l1_new, real_t l1_old,
        real_t w_mult
    ) nogil
    void predict_multiple(
        real_t *out,
        real_t *A, real_t *B,
        sparse_ix *ixA, sparse_ix *ixB,
        size_t n, int k,
        int nthreads
    ) nogil
    long double eval_llk(
        real_t *A,
        real_t *B,
        sparse_ix ixA[],
        sparse_ix ixB[],
        real_t *X,
        size_t nnz, int k,
        bint full_llk, bint include_missing,
        size_t dimA, size_t dimB,
        int nthreads
    ) nogil
    int factors_multiple(
        real_t *A, real_t *B,
        real_t *Bsum, real_t *Amean,
        real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
        int k, size_t dimA,
        real_t l2_reg, real_t w_mult,
        real_t step_size, size_t niter, size_t maxupd,
        Method method, bint limit_step, bint reuse_mean,
        int nthreads
    ) nogil
    int topN(
        real_t *a_vec, real_t *B, int k,
        sparse_ix *include_ix, size_t n_include,
        sparse_ix *exclude_ix, size_t n_exclude,
        sparse_ix *outp_ix, real_t *outp_score,
        size_t n_top, size_t n, int nthreads
    ) nogil

def _run_poismf(
    np.ndarray[real_t, ndim=1] Xr,
    np.ndarray[size_t, ndim=1] Xr_indices,
    np.ndarray[size_t, ndim=1] Xr_indptr,
    np.ndarray[real_t, ndim=1] Xc,
    np.ndarray[size_t, ndim=1] Xc_indices,
    np.ndarray[size_t, ndim=1] Xc_indptr,
    np.ndarray[real_t, ndim=2] A,
    np.ndarray[real_t, ndim=2] B,
    method="tncg", bint limit_step=0,
    real_t l2_reg=1e9, real_t l1_reg=0, real_t w_mult=1.,
    real_t step_size=1e-7,
    size_t niter=10, size_t maxupd=1,
    bint early_stop=1, bint reuse_prev=1,
    bint handle_interrupt=1,
    int nthreads=1):

    if Xr.shape[0] == 0:
        raise ValueError("'X' contains no non-zero entries.")

    ### Check for potential integer overflow
    INT_MAX = np.iinfo(ctypes.c_int).max
    if max(A.shape[0], A.shape[1], B.shape[0]) > INT_MAX:
        raise ValueError("Error: integer overflow. Dimensions cannot be larger than 2^31-1.")

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

    cdef int ret_code = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        ret_code = run_poismf(
            &A[0,0], &Xr[0], &Xr_indptr[0], &Xr_indices[0],
            &B[0,0], &Xc[0], &Xc_indptr[0], &Xc_indices[0],
            dimA, dimB, k,
            l2_reg, l1_reg, w_mult, step_size,
            c_method, limit_step, niter, maxupd,
            early_stop, reuse_prev,
            handle_interrupt, nthreads
        )
    if ret_code == 1:
        raise MemoryError("Could not allocate enough memory.")
    elif (ret_code == 2) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

def _predict_multiple(np.ndarray[real_t, ndim=1] out, np.ndarray[real_t, ndim=2] A, np.ndarray[real_t, ndim=2] B,
                      np.ndarray[size_t, ndim=1] ix_u, np.ndarray[size_t, ndim=1] ix_i, int nthreads):
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        predict_multiple(&out[0], &A[0,0], &B[0,0], &ix_u[0], &ix_i[0], ix_u.shape[0], A.shape[1], nthreads)

def _predict_factors(
        np.ndarray[real_t, ndim=1] counts,
        np.ndarray[size_t, ndim=1] ix,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=1] Bsum,
        np.ndarray[real_t, ndim=1] Amean,
        bint reuse_mean = 1,
        size_t maxupd = 20,
        real_t l2_reg = 1e5,
        real_t l1_new = 0., real_t l1_old = 0.,
        real_t w_mult = 1., bint limit_step = 0,
    ):
    cdef np.ndarray[real_t, ndim=1] out = np.empty(Amean.shape[0], dtype=c_real)
    cdef real_t *ptr_counts = NULL
    cdef size_t *ptr_ix = NULL
    if counts.shape[0]:
        ptr_counts = &counts[0]
    if ix.shape[0]:
        ptr_ix = &ix[0]
    cdef int ret_code = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        ret_code = factors_single(
            &out[0], <size_t> out.shape[0],
            &Amean[0], reuse_mean,
            ptr_counts, ptr_ix, <size_t> counts.shape[0],
            &B[0,0], &Bsum[0],
            maxupd, l2_reg, l1_new, l1_old,
            w_mult
        )
    if ret_code != 0:
        raise MemoryError("Could not allocate enough memory.")
    return out

def _predict_factors_multiple(
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=1] Bsum,
        np.ndarray[real_t, ndim=1] Amean,
        np.ndarray[size_t, ndim=1] Xr_indptr,
        np.ndarray[size_t, ndim=1] Xr_indices,
        np.ndarray[real_t, ndim=1] Xr,
        real_t l2_reg = 1e9,
        real_t w_mult = 1.,
        real_t step_size = 1e-7,
        size_t niter = 10,
        size_t maxupd = 1,
        method = "tncg",
        bint limit_step = 0,
        bint reuse_mean = 1,
        int nthreads = 1
    ):
    cdef int k = <size_t> B.shape[1]
    cdef size_t dimA = Xr_indptr.shape[0] - 1
    cdef np.ndarray[real_t, ndim=2] A = np.empty((dimA, k), dtype=c_real)

    cdef real_t *ptr_A = &A[0,0]
    cdef real_t *ptr_B = &B[0,0]
    cdef real_t *ptr_Bsum = &Bsum[0]
    cdef real_t *ptr_Amean = &Amean[0]
    cdef size_t *ptr_Xr_indptr = &Xr_indptr[0]
    
    cdef size_t *ptr_Xr_indices = NULL
    cdef real_t *ptr_Xr = NULL

    if Xr_indices.shape[0]:
        ptr_Xr_indices = &Xr_indices[0]
    if Xr.shape[0]:
        ptr_Xr = &Xr[0]

    cdef Method c_method = tncg
    if method == "tncg":
        c_method = tncg
    elif method == "cg":
        c_method = cg
    elif method == "pg":
        c_method = pg

    cdef int returned_val = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        returned_val = factors_multiple(
            ptr_A, ptr_B,
            ptr_Bsum, ptr_Amean,
            ptr_Xr, ptr_Xr_indptr, ptr_Xr_indices,
            k, dimA,
            l2_reg, w_mult,
            step_size, niter, maxupd,
            c_method, limit_step, reuse_mean,
            nthreads
        )

    if returned_val:
        raise MemoryError("Could not allocate enough memory.")

    return A

def _eval_llk_test(
        np.ndarray[real_t, ndim=2] A,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[size_t, ndim=1] ixA,
        np.ndarray[size_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        bint full_llk = 0, bint include_missing = 0,
        int nthreads = 1
    ):
    if ixA.shape[0] == 0:
        return 0.
    cdef real_t *ptr_A = &A[0,0]
    cdef real_t *ptr_B = &B[0,0]
    cdef size_t *ptr_ixA = &ixA[0]
    cdef size_t *ptr_ixB = &ixB[0]
    cdef real_t *ptr_X = &X[0]
    cdef size_t nnz = X.shape[0]
    cdef size_t dimA = A.shape[0]
    cdef size_t dimB = B.shape[0]
    cdef int k = A.shape[1]
    cdef long double res = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        res = eval_llk(
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
        np.ndarray[real_t, ndim=1] a_vec,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[size_t, ndim=1] include_ix,
        np.ndarray[size_t, ndim=1] exclude_ix,
        size_t top_n = 10,
        bint output_score = 0,
        int nthreads = 1
    ):
    cdef real_t *ptr_a = &a_vec[0]
    cdef real_t *ptr_B = &B[0,0]
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
    cdef np.ndarray[real_t, ndim=1] outp_score = np.empty(0, dtype=c_real)
    cdef size_t *ptr_outp_ix = &outp_ix[0]
    cdef real_t *ptr_outp_score = NULL
    if output_score:
        outp_score = np.empty(top_n, dtype=c_real)
        ptr_outp_score = &outp_score[0]

    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        topN(
            ptr_a, ptr_B, k,
            ptr_include, n_include,
            ptr_exclude, n_exclude,
            ptr_outp_ix, ptr_outp_score,
            top_n, n, nthreads
        )
    return outp_ix, outp_score

###################
#### Cblas Wrappers
#### Do not move to a new file as otherwise it doesn't compile,
#### due to generating duplicated file names between headers and C files
from scipy.linalg.cython_blas cimport ddot, daxpy, dscal, dnrm2, dgemv
from scipy.linalg.cython_blas cimport sdot, saxpy, sscal, snrm2, sgemv

ctypedef double (*ddot_)(const int*, const double*, const int*, const double*, const int*) nogil
ctypedef void (*daxpy_)(const int*, const double*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dscal_)(const int*, const double*, double*, const int*) nogil
ctypedef double (*dnrm2_)(const int*, const double*, const int*) nogil
ctypedef void (*dgemv_)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil

ctypedef float (*sdot_)(const int*, const float*, const int*, const float*, const int*) nogil
ctypedef void (*saxpy_)(const int*, const float*, const float*, const int*, float*, const int*) nogil
ctypedef void (*sscal_)(const int*, const float*, float*, const int*) nogil
ctypedef float (*snrm2_)(const int*, const float*, const int*) nogil
ctypedef void (*sgemv_)(const char*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*) nogil

ctypedef enum CBLAS_ORDER:
    CblasRowMajor = 101
    CblasColMajor = 102

ctypedef CBLAS_ORDER CBLAS_LAYOUT

ctypedef enum CBLAS_TRANSPOSE:
    CblasNoTrans=111
    CblasTrans=112
    CblasConjTrans=113
    CblasConjNoTrans=114

ctypedef enum CBLAS_UPLO:
    CblasUpper=121
    CblasLower=122

ctypedef enum CBLAS_DIAG:
    CblasNonUnit=131
    CblasUnit=132

ctypedef enum CBLAS_SIDE:
    CblasLeft=141
    CblasRight=142

cdef public double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) nogil:
    return (<ddot_>ddot)(&n, x, &incx, y, &incy)

cdef public void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) nogil:
    (<daxpy_>daxpy)(&n, &alpha, x, &incx, y, &incy)

cdef public void cblas_dscal(const int N, const double alpha, double *X, const int incX) nogil:
    (<dscal_>dscal)(&N, &alpha, X, &incX)

cdef public double cblas_dnrm2(const int n, const double *x, const int incx) nogil:
    return (<dnrm2_>dnrm2)(&n, x, &incx)

### Note: Cython refuses to compile a public cdef'd function with enum arguments
cdef public void cblas_dgemv(const int order,  const int TransA,  const int m, const int n,
     const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy) nogil:
    cdef char trans
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 'N';
        elif (TransA == CblasTrans):
            trans = 'T'
        else:
            trans = 'C'
        (<dgemv_>dgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy)

    else:
        if (TransA == CblasNoTrans):
            trans = 'T'
        elif (TransA == CblasTrans):
            trans = 'N'
        else:
            trans = 'N'

        (<dgemv_>dgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy)

##################

cdef public float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy) nogil:
    return (<sdot_>sdot)(&n, x, &incx, y, &incy)

cdef public void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy) nogil:
    (<saxpy_>saxpy)(&n, &alpha, x, &incx, y, &incy)

cdef public void cblas_sscal(const int N, const float alpha, float *X, const int incX) nogil:
    (<sscal_>sscal)(&N, &alpha, X, &incX)

cdef public float cblas_snrm2(const int n, const float *x, const int incx) nogil:
    return (<snrm2_>snrm2)(&n, x, &incx)

### Note: Cython refuses to compile a public cdef'd function with enum arguments
cdef public void cblas_sgemv(const int order,  const int TransA,  const int m, const int n,
     const float alpha, const float  *a, const int lda,  const float  *x, const int incx,  const float beta,  float  *y, const int incy) nogil:
    cdef char trans
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 'N';
        elif (TransA == CblasTrans):
            trans = 'T'
        else:
            trans = 'C'
        (<sgemv_>sgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy)

    else:
        if (TransA == CblasNoTrans):
            trans = 'T'
        elif (TransA == CblasTrans):
            trans = 'N'
        else:
            trans = 'N'

        (<sgemv_>sgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy)


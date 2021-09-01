 /*
    Poisson Factorization for sparse matrices

    Based on alternating proximal gradient iteration or conjugate gradient.
    Variables must be initialized from outside the main function ('run_poismf').
    Writen for C99 standard and OpenMP 2.0 or later.

    Reference paper is:
        Cortes, David.
        "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations."
        arXiv preprint arXiv:1811.01908 (2018).

    BSD 2-Clause License

    Copyright (c) 2018-2021, David Cortes
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/* Note: this file is a wrapper for the R language. It doesn't need to be compiled
   if wrapping it for a different language. */
#ifdef _FOR_R

#include "poismf.h"

/* FORTRAN-BLAS -> CBLAS */
#include <R_ext/BLAS.h>
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) { return ddot_(&n, x, &incx, y, &incy); }
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) { daxpy_(&n, &alpha, x, &incx, y, &incy); }
void cblas_dscal(const int N, const double alpha, double *X, const int incX) { dscal_(&N, &alpha, X, &incX); }
double cblas_dnrm2(const int n, const double *x, const int incx) { return dnrm2_(&n, x, &incx); }
void cblas_dgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE TransA,  const int m, const int n,
         const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy)
{
    char trans = '\0';
    if (order == CblasColMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 'N';
        else if (TransA == CblasTrans)
            trans = 'T';
        else
            trans = 'C';

        dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    else
    {
        if (TransA == CblasNoTrans)
            trans = 'T';
        else if (TransA == CblasTrans)
            trans = 'N';
        else
            trans = 'N';

        dgemv_(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
}

SEXP wrapper_run_poismf
(
    SEXP Xr, SEXP Xr_indices, SEXP Xr_indptr,
    SEXP Xc, SEXP Xc_indices, SEXP Xc_indptr,
    SEXP A, SEXP B, SEXP dimA, SEXP dimB, SEXP k,
    SEXP method, SEXP limit_step,
    SEXP l2_reg, SEXP l1_reg,
    SEXP w_mult, SEXP step_size,
    SEXP niter, SEXP maxupd,
    SEXP early_stop, SEXP reuse_prev,
    SEXP handle_interrupt, SEXP nthreads
)
{
    if (Rf_xlength(Xr) == 0) {
        Rf_error("'X' contains no non-zero entries.");
        return R_NilValue;
    }

    int ret_code = run_poismf(
        REAL(A), REAL(Xr), INTEGER(Xr_indptr), INTEGER(Xr_indices),
        REAL(B), REAL(Xc), INTEGER(Xc_indptr), INTEGER(Xc_indices),
        (size_t) Rf_asInteger(dimA), (size_t) Rf_asInteger(dimB),
        (size_t) Rf_asInteger(k), Rf_asReal(l2_reg), Rf_asReal(l1_reg),
        Rf_asReal(w_mult), Rf_asReal(step_size),
        (Method) Rf_asInteger(method), (bool) Rf_asLogical(limit_step),
        (size_t) Rf_asInteger(niter), (size_t) Rf_asInteger(maxupd),
        (bool) Rf_asLogical(early_stop), (bool) Rf_asLogical(reuse_prev),
        (bool) Rf_asLogical(handle_interrupt), Rf_asInteger(nthreads)
    );
    
    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();
    if (ret_code == 1)
        Rf_error("Out of memory.");
    else if (ret_code == 2 && !((bool) Rf_asLogical(handle_interrupt)))
        Rf_error("Procedure was interrupted.");
    return R_NilValue;
}

SEXP wrapper_predict_multiple
(
    SEXP A, SEXP B, SEXP k,
    SEXP ixA, SEXP ixB,
    SEXP nthreads
)
{
    size_t nnz = (size_t) Rf_xlength(ixA);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, nnz));
    predict_multiple(
        REAL(out), REAL(A), REAL(B),
        INTEGER(ixA), INTEGER(ixB),
        nnz, Rf_asInteger(k),
        Rf_asInteger(nthreads)
    );
    UNPROTECT(1);
    return out;
}

SEXP wrapper_predict_factors
(
    SEXP k, SEXP Amean, SEXP reuse_mean,
    SEXP counts, SEXP ix,
    SEXP B, SEXP Bsum,
    SEXP maxupd,
    SEXP l2_reg,
    SEXP l1_new, SEXP l1_old,
    SEXP w_mult
)
{
    size_t k_szt = (size_t) Rf_asInteger(k);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, k_szt));

    int ret_code = factors_single(
        REAL(out), k_szt,
        REAL(Amean), (bool) Rf_asLogical(reuse_mean),
        REAL(counts), INTEGER(ix),
        (size_t) Rf_xlength(counts),
        REAL(B), REAL(Bsum),
        Rf_asInteger(maxupd),
        Rf_asReal(l2_reg),
        Rf_asReal(l1_new), Rf_asReal(l1_old),
        Rf_asReal(w_mult)
    );

    UNPROTECT(1);
    if (ret_code != 0) {
        Rf_error("Out of memory.");
        return R_NilValue; /* not reached */
    }
    return out;
}

SEXP wrapper_predict_factors_multiple
(
    SEXP dimA, SEXP k,
    SEXP B, SEXP Bsum, SEXP Amean,
    SEXP Xr_indptr, SEXP Xr_indices, SEXP Xr,
    SEXP l2_reg, SEXP w_mult,
    SEXP step_size, SEXP niter, SEXP maxupd,
    SEXP method, SEXP limit_step,
    SEXP reuse_mean, SEXP nthreads
)
{
    if ((R_xlen_t)Rf_asInteger(k) * (R_xlen_t)Rf_asInteger(dimA) <= 0)
    {
        Rf_error("Requested array dimensions exceed R limits.");
        return R_NilValue; /* not reached */
    }
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)Rf_asInteger(k)
                                                * (R_xlen_t)Rf_asInteger(dimA)));

    int res = factors_multiple(
        REAL(out), REAL(B),
        REAL(Bsum), REAL(Amean),
        REAL(Xr), INTEGER(Xr_indptr), INTEGER(Xr_indices),
        Rf_asInteger(k), Rf_asInteger(dimA),
        Rf_asReal(l2_reg), Rf_asReal(w_mult),
        Rf_asReal(step_size), (size_t) Rf_asInteger(niter),
        (size_t) Rf_asInteger(maxupd),
        (Method) Rf_asInteger(method),
        (bool) Rf_asLogical(limit_step),
        (bool) Rf_asLogical(reuse_mean),
        Rf_asInteger(nthreads)
    );

    UNPROTECT(1);
    if (res != 0) {
        Rf_error("Out of memory.");
        return R_NilValue; /* not reached */
    }
    return out;
}

SEXP wrapper_eval_llk
(
    SEXP A, SEXP B, SEXP dimA, SEXP dimB, SEXP k,
    SEXP ixA, SEXP ixB, SEXP Xcoo,
    SEXP full_llk, SEXP include_missing,
    SEXP nthreads
)
{
    long double llk = eval_llk(
        REAL(A), REAL(B),
        INTEGER(ixA), INTEGER(ixB), REAL(Xcoo),
        (size_t) Rf_xlength(Xcoo), Rf_asInteger(k),
        (bool) Rf_asLogical(full_llk), (bool) Rf_asLogical(include_missing),
        (size_t) Rf_asInteger(dimA), (size_t) Rf_asInteger(dimB),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(REALSXP, 1));
    REAL(out)[0] = (double)llk;
    UNPROTECT(1);
    return out;
}

SEXP wrapper_topN
(
    SEXP outp_ix, SEXP outp_score,
    SEXP a_vec, SEXP B, SEXP dimB,
    SEXP include_ix, SEXP exclude_ix,
    SEXP top_n, SEXP nthreads
)
{
    size_t n_include = (size_t) Rf_xlength(include_ix);
    size_t n_exclude = (size_t) Rf_xlength(exclude_ix);
    int res = topN(
        REAL(a_vec), REAL(B), Rf_xlength(a_vec),
        n_include? INTEGER(include_ix) : (int*)NULL, n_include,
        n_exclude? INTEGER(exclude_ix) : (int*)NULL, n_exclude,
        INTEGER(outp_ix),
        (Rf_xlength(outp_score) > 0)?
            REAL(outp_score) : (double*)NULL,
        (size_t) Rf_asInteger(top_n), (size_t) Rf_asInteger(dimB),
        Rf_asInteger(nthreads)
    );
    
    if (res != 0) /* out-of-memory */
        Rf_error("Out of memory.");

    return R_NilValue;
}

SEXP check_size_below_int_max
(
    SEXP dim1, SEXP dim2
)
{
    SEXP out = PROTECT(Rf_allocVector(LGLSXP, 1));
    LOGICAL(out)[0] = (size_t)Rf_asInteger(dim1) * (size_t)Rf_asInteger(dim2) <= INT_MAX;
    UNPROTECT(1);
    return out;
}

SEXP initialize_factors_mat(SEXP dim1, SEXP dim2)
{
    /* This is the initialization that was used in the original HPF code */
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, Rf_asInteger(dim1), Rf_asInteger(dim2)));
    double *restrict ptr_mat = REAL(out);
    size_t tot_size = (size_t)Rf_asInteger(dim1) * (size_t)Rf_asInteger(dim2);
    GetRNGstate();
    for (size_t ix = 0; ix < tot_size; ix++)
        ptr_mat[ix] = unif_rand();
    PutRNGstate();
    for (size_t ix = 0; ix < tot_size; ix++)
        ptr_mat[ix] = 0.3 + ptr_mat[ix] / 100.;
    UNPROTECT(1);
    return out;
}

SEXP R_has_openmp()
{
    #ifdef _OPENMP
    return Rf_ScalarLogical(1);
    #else
    return Rf_ScalarLogical(0);
    #endif
}

static const R_CallMethodDef callMethods [] = {
    {"wrapper_run_poismf", (DL_FUNC) &wrapper_run_poismf, 23},
    {"wrapper_predict_multiple", (DL_FUNC) &wrapper_predict_multiple, 6},
    {"wrapper_predict_factors", (DL_FUNC) &wrapper_predict_factors, 12},
    {"wrapper_predict_factors_multiple", (DL_FUNC) &wrapper_predict_factors_multiple, 17},
    {"wrapper_eval_llk", (DL_FUNC) &wrapper_eval_llk, 11},
    {"wrapper_topN", (DL_FUNC) &wrapper_topN, 9},
    {"check_size_below_int_max", (DL_FUNC) &check_size_below_int_max, 2},
    {"initialize_factors_mat", (DL_FUNC) &initialize_factors_mat, 2},
    {"R_has_openmp", (DL_FUNC) &R_has_openmp, 0},
    {NULL, NULL, 0}
}; 

void attribute_visible R_init_poismf(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}

#endif /* _FOR_R */

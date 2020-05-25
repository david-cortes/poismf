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

    Copyright (c) 2020, David Cortes
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
#include <R_ext/Rdynload.h>
#include <R.h>
#include <Rinternals.h>
#include "poismf.h"

/* FORTRAN-BLAS -> CBLAS */
#include <R_ext/BLAS.h>
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) { return ddot_(&n, x, &incx, y, &incy); }
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) { daxpy_(&n, &alpha, x, &incx, y, &incy); }
void cblas_dscal(const int N, const double alpha, double *X, const int incX) { dscal_(&N, &alpha, X, &incX); }
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
    SEXP use_cg, SEXP limit_step,
    SEXP l2_reg, SEXP l1_reg,
    SEXP w_mult, SEXP step_size,
    SEXP niter, SEXP npass, SEXP nthreads
)
{
    int ret_code = run_poismf(
        REAL(A), REAL(Xr), INTEGER(Xr_indptr), INTEGER(Xr_indices),
        REAL(B), REAL(Xc), INTEGER(Xc_indptr), INTEGER(Xc_indices),
        (size_t) Rf_asInteger(dimA), (size_t) Rf_asInteger(dimB),
        (size_t) Rf_asInteger(k), Rf_asReal(l2_reg), Rf_asReal(l1_reg),
        Rf_asReal(w_mult), Rf_asReal(step_size),
        (bool) Rf_asLogical(use_cg), (bool) Rf_asLogical(limit_step),
        (size_t) Rf_asInteger(niter), (size_t) Rf_asInteger(npass),
        Rf_asInteger(nthreads)
    );
    if (ret_code == 1) Rf_error("Out of memory.");
    return R_NilValue;
}

SEXP wrapper_predict_multiple
(
    SEXP A, SEXP B, SEXP k,
    SEXP ixA, SEXP ixB,
    SEXP nthreads
)
{
    size_t nnz = (size_t) Rf_length(ixA);
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
    SEXP A_old, SEXP k,
    SEXP counts, SEXP ix,
    SEXP B, SEXP Bsum,
    SEXP npass,
    SEXP l2_reg,
    SEXP l1_new, SEXP l1_old,
    SEXP w_mult, SEXP limit_step
)
{
    size_t k_szt = (size_t) Rf_asInteger(k);
    size_t dimA = (size_t)Rf_length(A_old) / k_szt;
    SEXP out = PROTECT(Rf_allocVector(REALSXP, k_szt));

    int ret_code = factors_single(
        REAL(out), k_szt,
        REAL(A_old), dimA,
        REAL(counts), INTEGER(ix),
        (sparse_ix) Rf_length(counts),
        REAL(B), REAL(Bsum),
        (size_t) Rf_asInteger(npass),
        Rf_asReal(l2_reg),
        Rf_asReal(l1_new), Rf_asReal(l1_old),
        Rf_asReal(w_mult),
        (bool) Rf_asLogical(limit_step)
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
    SEXP A_old, SEXP dimA, SEXP k,
    SEXP B, SEXP Bsum,
    SEXP Xr_indptr, SEXP Xr_indices, SEXP Xr,
    SEXP l2_reg, SEXP w_mult,
    SEXP step_size, SEXP niter, SEXP npass,
    SEXP use_cg, SEXP limit_step, SEXP nthreads
)
{
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (size_t)Rf_asInteger(k)
                                                * (size_t)Rf_asInteger(dimA)));

    int res = factors_multiple(
        REAL(out), REAL(B), REAL(A_old), REAL(Bsum),
        REAL(Xr), INTEGER(Xr_indptr), INTEGER(Xr_indices),
        Rf_asInteger(k), Rf_asInteger(dimA),
        Rf_asReal(l2_reg), Rf_asReal(w_mult),
        Rf_asReal(step_size), (size_t) Rf_asInteger(niter),
        (size_t) Rf_asInteger(npass),
        (bool) Rf_asLogical(use_cg),
        (bool) Rf_asLogical(limit_step),
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
        (size_t) Rf_length(Xcoo), Rf_asInteger(k),
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
    size_t n_include = (size_t) Rf_length(include_ix);
    size_t n_exclude = (size_t) Rf_length(exclude_ix);
    int res = topN(
        REAL(a_vec), REAL(B), Rf_length(a_vec),
        n_include? INTEGER(include_ix) : (int*)NULL, n_include,
        n_exclude? INTEGER(exclude_ix) : (int*)NULL, n_exclude,
        INTEGER(outp_ix),
        (Rf_length(outp_score) > 0)?
            REAL(outp_score) : (double*)NULL,
        (size_t) Rf_asInteger(top_n), (size_t) Rf_asInteger(dimB),
        Rf_asInteger(nthreads)
    );
    
    if (res != 0) /* out-of-memory */
        Rf_error("Out of memory.");

    return R_NilValue;
}



static const R_CallMethodDef callMethods [] = {
    {"wrapper_run_poismf", (DL_FUNC) &wrapper_run_poismf, 20},
    {"wrapper_predict_multiple", (DL_FUNC) &wrapper_predict_multiple, 6},
    {"wrapper_predict_factors", (DL_FUNC) &wrapper_predict_factors, 12},
    {"wrapper_predict_factors_multiple", (DL_FUNC) &wrapper_predict_factors_multiple, 16},
    {"wrapper_eval_llk", (DL_FUNC) &wrapper_eval_llk, 11},
    {"wrapper_topN", (DL_FUNC) &wrapper_topN, 9},
    {NULL, NULL, 0}
}; 

void R_init_poismf(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}


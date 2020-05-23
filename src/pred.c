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
#include "poismf.h"

/* Generic helper function that predicts multiple combinations of users and items from already-fit A and B matrices */
void predict_multiple
(
    double *restrict out,
    double *restrict A, double *restrict B,
    sparse_ix *ixA, sparse_ix *ixB,
    size_t n, int k,
    int nthreads
)
{
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix;
    #else
    size_t ix;
    #endif

    size_t k_szt = (size_t) k;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(out, n, A, B, ixA, ixB, k, k_szt)
    for (ix = 0; ix < n; ix++) {
        out[ix] = cblas_ddot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                B + (size_t)ixB[ix] * k_szt, 1);
    }
}

long double eval_llk
(
    double *restrict A,
    double *restrict B,
    sparse_ix ixA[],
    sparse_ix ixB[],
    double *restrict X,
    size_t nnz, int k,
    bool full_llk, bool include_missing,
    size_t dimA, size_t dimB,
    int nthreads
)
{
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix;
    #else
    size_t ix;
    #endif
    double pred;
    size_t k_szt = (size_t)k;
    long double llk = 0;

    double *restrict sumA = NULL;
    double *restrict sumB = NULL;

    if (!include_missing)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                reduction(+:llk) private(pred) shared(A, B, ixA, ixB, k, k_szt, X, nnz)
        for (ix = 0; ix < nnz; ix++)
        {
            pred = cblas_ddot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                 B + (size_t)ixB[ix] * k_szt, 1);
            llk += X[ix] * log(pred) - pred;
        }
    }

    else
    {
        sumA = (double*) malloc(k_szt * sizeof(double));
        sumB = (double*) malloc(k_szt * sizeof(double));
        if (sumA == NULL || sumB == NULL) {
            fprintf(stderr, "Error: out of memory.\n");
            llk = NAN;
            goto cleanup;
        }

        sum_by_cols(sumA, A, dimA, k_szt);
        sum_by_cols(sumB, B, dimB, k_szt);
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                reduction(+:llk) shared(X, nnz, A, B, ixA, ixB, k, k_szt)
        for (ix = 0; ix < nnz; ix++)
            llk += X[ix] * log(cblas_ddot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                             B + (size_t)ixB[ix] * k_szt, 1));
        llk -= cblas_ddot(k, sumA, 1, sumB, 1);
    }

    if (full_llk)
    {
        long double llk_const = 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                reduction(+:llk_const) shared(X, nnz)
        for (ix = 0; ix < nnz; ix++)
            llk_const += lgamma(X[ix] + 1.);
        llk -= llk_const;
    }

    cleanup:
        free(sumA);
        free(sumB);
    return llk;
} 

int factors_multiple
(
    double *A, double *B, double *A_old, double *Bsum,
    double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    int k, size_t dimA,
    double l2_reg, double w_mult,
    double step_size, size_t niter, size_t npass,
    bool use_cg,
    int nthreads
)
{
    /* Note: Bsum should already have the l1 regularization added to it */
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix;
    #else
    size_t ix;
    #endif

    size_t k_szt = (size_t) k;
    double cnst_div;
    double *Bsum_w = NULL;
    double *buffer_arr = (double*) malloc(k_szt * sizeof(double) * (size_t)nthreads
                                          * (use_cg? 5 : 1));

    int return_val = 0;

    if (buffer_arr == NULL) {
        throw_oom:
            fprintf(stderr, "Error: out of memory.\n");
            return_val = 1;
            goto cleanup;
    }

    if (w_mult != 1.) {
        Bsum_w = (double*)malloc(sizeof(double) * k * dimA);
        if (Bsum_w == NULL) goto throw_oom;
        adjustment_Bsum(B, Bsum, Bsum_w,
                        Xr_indices, Xr_indptr,
                        dimA, k, w_mult, nthreads);
    }

    /* Initialize all values to the mean of old A */
    sum_by_cols(A, A_old, dimA, k_szt);
    cblas_dscal(k, 1./(double)dimA, A, 1);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(dimA, A, k_szt)
    for (ix = 1; ix < dimA; ix++)
        memcpy(A + (size_t)ix*k_szt, A, k_szt*sizeof(double));

    if (use_cg)
    {
        cg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                     dimA, k_szt,
                     Bsum, l2_reg, w_mult, npass * niter,
                     buffer_arr, Bsum_w, nthreads);
    }

    else
    {
        for (size_t iter = 0; iter < niter; iter++)
        {
            cnst_div = 1. / (1. + 2. * l2_reg * step_size);
            pg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                         dimA, k, cnst_div, Bsum, step_size,
                         w_mult, npass, buffer_arr, nthreads);
            step_size *= 0.5;
        }
    }

    cleanup:
        free(Bsum_w);
        free(buffer_arr);
    return return_val;
}

int factors_single
(
    double *restrict out, size_t k,
    double *restrict A_old, size_t dimA,
    double *restrict X, sparse_ix X_ind[], size_t nnz,
    double *restrict B, double *restrict Bsum,
    size_t npass, double l2_reg, double l1_new, double l1_old,
    double w_mult
)
{
    /* Note: Bsum should already have the *old* l1 regularization added to it */
    int k_int = (int) k;
    double l1_reg = l1_new - l1_old;
    double *restrict Bsum_pass = NULL;
    if (l1_reg > 0. || w_mult != 1.)
    {
        Bsum_pass = (double*)malloc(sizeof(double) * k);
        if (Bsum_pass == NULL) {
            fprintf(stderr, "Error: out of memory.\n");
            return 1;
        }

        if (w_mult != 1.) {
            memset(Bsum_pass, 0, sizeof(double) * k);
            for (size_t ix = 0; ix < nnz; ix++)
                cblas_daxpy(k_int, 1., B + X_ind[ix]*k, 1, Bsum_pass, 1);
            cblas_dscal(k_int, w_mult - 1., Bsum_pass, 1);
            cblas_daxpy(k_int, 1., Bsum, 1, Bsum_pass, 1);
        }

        else {
            memcpy(Bsum_pass, Bsum, sizeof(double) * k);
        }

        if (l1_reg > 0.) {
            for (size_t ix = 0; ix < k; ix++)
                Bsum_pass[ix] += l1_reg;
        }
    }

    else {
        Bsum_pass = Bsum;
    }

    /* Initialize to the mean of current factors */
    sum_by_cols(out, A_old, dimA, k);
    cblas_dscal((int)k, 1./(double)dimA, out, 1);

    fdata data = { B, Bsum_pass, X, X_ind, nnz, l2_reg, w_mult };
    double fun_val;
    size_t niter;
    size_t nfeval;

    int ret_code = minimize_nonneg_cg(
        out, (int)k, &fun_val,
        calc_fun_single, calc_grad_single, NULL, (void*) &data,
        1e-5, 250, npass, &niter, &nfeval,
        0.25, 0.01, 20,
        NULL, 1, 0);

    if (l1_reg > 0. || w_mult != 1.) free(Bsum_pass);
    if (ret_code == 4) {
        fprintf(stderr, "Error: out of memory.\n");
        return 1;
    } else {
        return 0;
    }
}

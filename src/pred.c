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
    real_t *restrict out,
    real_t *restrict A, real_t *restrict B,
    sparse_ix *ixA, sparse_ix *ixB,
    size_t n, int k,
    int nthreads
)
{
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix = 0;
    #else
    size_t ix = 0;
    #endif

    size_t k_szt = (size_t) k;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(out, n, A, B, ixA, ixB, k, k_szt)
    for (ix = 0; ix < n; ix++) {
        out[ix] = cblas_tdot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                B + (size_t)ixB[ix] * k_szt, 1);
    }
}

long double eval_llk
(
    real_t *restrict A,
    real_t *restrict B,
    sparse_ix ixA[],
    sparse_ix ixB[],
    real_t *restrict X,
    size_t nnz, int k,
    bool full_llk, bool include_missing,
    size_t dimA, size_t dimB,
    int nthreads
)
{
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix = 0;
    #else
    size_t ix = 0;
    #endif
    real_t pred;
    size_t k_szt = (size_t)k;
    long double llk = 0;

    real_t *restrict sumA = NULL;
    real_t *restrict sumB = NULL;

    if (!include_missing)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                reduction(+:llk) private(pred) shared(A, B, ixA, ixB, k, k_szt, X, nnz)
        for (ix = 0; ix < nnz; ix++)
        {
            pred = cblas_tdot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                 B + (size_t)ixB[ix] * k_szt, 1);
            llk += X[ix] * log(pred) - pred;
        }
    }

    else
    {
        sumA = (real_t*) malloc(k_szt * sizeof(real_t));
        sumB = (real_t*) malloc(k_szt * sizeof(real_t));
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
            llk += X[ix] * log(cblas_tdot(k, A + (size_t)ixA[ix] * k_szt, 1,
                                             B + (size_t)ixB[ix] * k_szt, 1));
        llk -= cblas_tdot(k, sumA, 1, sumB, 1);
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
    real_t *A, real_t *B, real_t *A_old, real_t *Bsum,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    int k, size_t dimA,
    real_t l2_reg, real_t w_mult,
    real_t step_size, size_t niter, size_t maxupd,
    Method method, bool limit_step,
    int nthreads
)
{
    /* Note: Bsum should already have the l1 regularization added to it */
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix = 0;
    #else
    size_t ix = 0;
    #endif

    size_t k_szt = (size_t) k;
    real_t cnst_div;
    real_t *Bsum_w = NULL;
    real_t *Bsum_w_scaled = NULL;
    real_t *Bsum_scaled = NULL;
    size_t size_buffer = 1;
    switch(method) {
        case pg:   {size_buffer = 1;  break;}
        case cg:   {size_buffer = 5;  break;}
        case tncg: {size_buffer = 22; break;}
    }
    size_buffer *= (k_szt * (size_t)nthreads);
    real_t *buffer_arr = (real_t*) malloc(sizeof(real_t) * size_buffer);
    int *buffer_int = NULL;
    real_t *zeros_tncg = NULL;
    real_t *inf_tncg = NULL;

    int return_val = 0;

    if (buffer_arr == NULL) {
        throw_oom:
            fprintf(stderr, "Error: out of memory.\n");
            return_val = 1;
            goto cleanup;
    }

    if (method == pg) {
        if (w_mult == 1.) {
            Bsum_scaled = (real_t*)malloc(sizeof(real_t) * k_szt);
            if (Bsum_scaled == NULL) goto throw_oom;
        }

        else {
            Bsum_w_scaled = (real_t*)malloc(sizeof(real_t) * k_szt * dimA);
            if (Bsum_w_scaled == NULL) goto throw_oom;
        }
    }

    if (w_mult != 1.) {
        Bsum_w = (real_t*)malloc(sizeof(real_t) * k_szt * dimA);
        if (Bsum_w == NULL) goto throw_oom;
        adjustment_Bsum(B, Bsum, Bsum_w,
                        Xr_indices, Xr_indptr,
                        dimA, k, w_mult, nthreads);
        if (method == pg)
            dscal_large(dimA*k_szt, -step_size, Bsum_w);
    }

    if (method == tncg) {
        buffer_int = (int*)malloc(sizeof(int) * k_szt * (size_t)nthreads);
        zeros_tncg = (real_t*)calloc(sizeof(real_t), k_szt);
        inf_tncg = (real_t*)malloc(sizeof(real_t) * k_szt);
        if (buffer_int == NULL || zeros_tncg == NULL || inf_tncg == NULL)
            goto throw_oom;
        for (size_t ix = 0; ix < k_szt; ix++)
            inf_tncg[ix] = HUGE_VAL;
    }

    /* Initialize all values to the mean of old A */
    sum_by_cols(A, A_old, dimA, k_szt);
    cblas_tscal(k, 1./(real_t)dimA, A, 1);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(dimA, A, k_szt)
    for (ix = 1; ix < dimA; ix++)
        memcpy(A + (size_t)ix*k_szt, A, k_szt*sizeof(real_t));

    switch(method) {
        case pg:
        {
            for (size_t iter = 0; iter < niter; iter++)
            {
                if (w_mult == 1.) {
                    memcpy(Bsum_scaled, Bsum, sizeof(real_t) * k_szt);
                    cblas_tscal(k, -step_size, Bsum_scaled, 1);
                }
                else {
                    memcpy(Bsum_w_scaled, Bsum_w, sizeof(real_t) * k_szt * dimA);
                    dscal_large(dimA*k_szt, -step_size, Bsum_w_scaled);
                }
                cnst_div = 1. / (1. + 2. * l2_reg * step_size);
                pg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                             dimA, k, cnst_div, Bsum_scaled, Bsum_w_scaled,
                             step_size, w_mult, maxupd, buffer_arr, nthreads);
                step_size *= 0.5;
            }
            break;
        }

        case cg:
        {
            cg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                         dimA, k_szt, limit_step,
                         Bsum, l2_reg, w_mult, maxupd * niter,
                         buffer_arr, Bsum_w, nthreads);
            break;
        }

        case tncg:
        {
            tncg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                           dimA, k_szt, Bsum, l2_reg, w_mult, maxupd,
                           buffer_arr, buffer_int,
                           zeros_tncg, inf_tncg,
                           Bsum_w, nthreads);
            break;
        }
    }

    cleanup:
        free(Bsum_w);
        free(Bsum_w_scaled);
        free(Bsum_scaled);
        free(buffer_arr);
        free(buffer_int);
    return return_val;
}

int factors_single
(
    real_t *restrict out, size_t k,
    real_t *restrict A_old, size_t dimA,
    real_t *restrict X, sparse_ix X_ind[], size_t nnz,
    real_t *restrict B, real_t *restrict Bsum,
    int maxupd, real_t l2_reg, real_t l1_new, real_t l1_old,
    real_t w_mult
)
{
    /* Note: Bsum should already have the *old* l1 regularization added to it */
    int k_int = (int) k;
    real_t l1_reg = l1_new - l1_old;
    real_t *restrict Bsum_pass = NULL;
    real_t *restrict zeros_tncg = (real_t*)calloc(sizeof(real_t), k);
    real_t *restrict inf_tncg = (real_t*)malloc(sizeof(real_t) * k);
    int ret_code = 0;

    fdata data = { B, NULL, X, X_ind, nnz, l2_reg, w_mult, k_int };
    real_t fun_val = 0;
    int niter = 0;
    int nfeval = 0;

    if (zeros_tncg == NULL || inf_tncg == NULL)
        goto throw_oom;

    if (l1_reg > 0. || w_mult != 1.)
    {
        Bsum_pass = (real_t*)malloc(sizeof(real_t) * k);
        if (Bsum_pass == NULL) {
            throw_oom:
                fprintf(stderr, "Error: out of memory.\n");
                ret_code = 1;
                goto cleanup;
        }

        if (w_mult != 1.) {
            memset(Bsum_pass, 0, sizeof(real_t) * k);
            for (size_t ix = 0; ix < nnz; ix++)
                cblas_taxpy(k_int, 1., B + X_ind[ix]*k, 1, Bsum_pass, 1);
            cblas_tscal(k_int, w_mult - 1., Bsum_pass, 1);
            cblas_taxpy(k_int, 1., Bsum, 1, Bsum_pass, 1);
        }

        else {
            memcpy(Bsum_pass, Bsum, sizeof(real_t) * k);
        }

        if (l1_reg > 0.) {
            for (size_t ix = 0; ix < k; ix++)
                Bsum_pass[ix] += l1_reg;
        }
    }

    else {
        Bsum_pass = Bsum;
    }

    data.Bsum = Bsum_pass;

    for (size_t ix = 0; ix < k; ix++)
        inf_tncg[ix] = HUGE_VAL;

    /* Initialize to the mean of current factors */
    sum_by_cols(out, A_old, dimA, k);
    cblas_tscal((int)k, 1./(real_t)dimA, out, 1);

    ret_code = tnc(
        k_int, out, &fun_val,
        (real_t*)NULL,
        calc_fun_and_grad,
        (void*) &data, zeros_tncg, inf_tncg, (real_t*)NULL,
        (real_t*)NULL, 0, -1, maxupd,
        0.25, 10., 0., 0.,
        1e-4, -1., -1., 1.3,
        &nfeval, &niter,
        (real_t*)NULL, (int*)NULL);

    if (ret_code == -3) {
        goto throw_oom;
    } else {
        ret_code = 0;
    }

    cleanup:
        if (l1_reg > 0. || w_mult != 1.)
            free(Bsum_pass);
        free(zeros_tncg);
        free(inf_tncg);
        return ret_code;
}

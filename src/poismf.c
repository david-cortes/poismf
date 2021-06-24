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
#include "poismf.h"


/* Interrupt handler */
bool should_stop_procedure = false;
bool handle_is_locked = false;
void set_interrup_global_variable(int s)
{
    #pragma omp critical
    {
        fprintf(stderr, "Error: procedure was interrupted\n");
        should_stop_procedure = true;
    }
}

/* Helper functions */
#define nonneg(x) (((x) > 0.)? (x) : 0.)

void dscal_large(size_t n, real_t alpha, real_t *restrict x)
{
    if (n < (size_t)INT_MAX)
        cblas_tscal((int)n, alpha, x, 1);
    else {
        for (size_t ix = 0; ix < n; ix++)
            x[ix] *= alpha;
    }
}

void sum_by_cols(real_t *restrict out, real_t *restrict M, size_t nrow, size_t ncol)
{
    memset(out, 0, sizeof(real_t) * ncol);
    for (size_t row = 0; row < nrow; row++)
        for (size_t col = 0; col < ncol; col++)
            out[col] += M[row*ncol + col];
}

void adjustment_Bsum
(
    real_t *restrict B,
    real_t *restrict Bsum,
    real_t *restrict Bsum_user,
    sparse_ix Xr_indices[],
    sparse_ix Xr_indptr[],
    size_t dimA, size_t k,
    real_t w_mult, int nthreads
)
{
    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64)) /* OpenMP < 3.0 */
    long long ix = 0;
    long long row = 0;
    #else
    size_t ix = 0;
    size_t row = 0;
    #endif

    int k_int = (int) k;
    memset(Bsum_user, 0, dimA*k*sizeof(real_t));
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(dimA, Xr_indptr, Xr_indices, B, Bsum_user, k_int)
    for (row = 0; row < dimA; row++)
        for (size_t ix = Xr_indptr[row]; ix < Xr_indptr[row + 1]; ix++)
            cblas_taxpy(k_int, 1., B + Xr_indices[ix]*k, 1, Bsum_user + row*k, 1);

    size_t n = dimA * k;
    real_t new_w = w_mult - 1.;
    /* Note: don't use daxpy here as 'n' might be larger than INT_MAX */
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(n, new_w, Bsum_user)
    for (ix = 0; ix < n; ix++)
        Bsum_user[ix] *= new_w;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(dimA, k, k_int, Bsum, Bsum_user)
    for (row = 0; row < dimA; row++)
        cblas_taxpy(k_int, 1., Bsum, 1, Bsum_user + row*k, 1);
}

/* Functions for Proximal Gradient */
void calc_grad_pgd(real_t *out, real_t *curr, real_t *F, real_t *X, sparse_ix *Xind, sparse_ix nnz_this, int k)
{
    size_t k_szt = (size_t)k;
    memset(out, 0, sizeof(real_t) * (size_t)k);
    for (sparse_ix ix = 0; ix < nnz_this; ix++)
        cblas_taxpy(k, X[ix] / cblas_tdot(k, F + (size_t)Xind[ix] * k_szt, 1, curr, 1),
                    F + (size_t)Xind[ix] * k_szt, 1, out, 1);
}

/*  This function is written having in mind the A matrix being optimized,
    with the B matrix being fixed, and the data passed in row-sparse format.
    For optimizing B, swap any mention of A and B, and pass the data in
    column-sparse format */
void pg_iteration
(
    real_t *A, real_t *B,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    real_t cnst_div, real_t *cnst_sum, real_t *Bsum_user,
    real_t step_size, real_t w_mult, size_t maxupd,
    real_t *buffer_arr, int nthreads
)
{
    int k_int = (int) k;
    sparse_ix nnz_this;
    step_size *= w_mult;

    real_t *Bsum = cnst_sum;

    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64)) /* OpenMP < 3.0 */
    long long ia;
    #endif

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            firstprivate(Bsum) private(nnz_this) \
            shared(A, B, k, k_int, cnst_div, Bsum_user, maxupd, Xr, Xr_indptr, Xr_indices)
    for (size_t_for ia = 0; ia < dimA; ia++)
    {

        nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];
        if (nnz_this == 0) {
            memset(A + ia*k, 0, k*sizeof(real_t));
            continue;
        }
        if (w_mult != 1.) Bsum = Bsum_user + ia*k;

        for (size_t p = 0; p < maxupd; p++)
        {
            calc_grad_pgd(buffer_arr + k*omp_get_thread_num(),
                          A + ia*k, B, Xr + Xr_indptr[ia],
                          Xr_indices + Xr_indptr[ia], nnz_this, k_int);
            cblas_taxpy(k_int, step_size,
                        buffer_arr + k*omp_get_thread_num(), 1,
                        A + ia*k, 1);

            cblas_taxpy(k_int, 1., Bsum, 1, A + ia*k, 1);
            cblas_tscal(k_int, cnst_div, A + ia*k, 1);
            for (size_t ix = 0; ix < k; ix++)
                A[ia*k + ix] = nonneg(A[ia*k + ix]);
        }

    }
}

/* Functions for Conjugate Gradient */
/* TODO: when doing line searches without evaluating gradient (e.g. for CG),
   this could be computed faster by keeping pt1=B*a_vec; pt2=B*alpha*grad,
   and then evaluating only loss(pt1+step*pt2).  */
void calc_fun_single(real_t a_row[], int k_int, real_t *f, void *data)
{
    fdata* fun_data = (fdata*) data;
    size_t k = (size_t)k_int;
    real_t reg_term = cblas_tdot(k_int, fun_data->Bsum, 1, a_row, 1);
    reg_term += fun_data->l2_reg * cblas_tdot(k_int, a_row, 1, a_row, 1);
    real_t lsum = 0.;
    for (size_t ix = 0; ix < fun_data->nnz_this; ix++)
    {
        lsum += fun_data->Xr[ix]
                 * log( cblas_tdot(k_int, a_row, 1,
                                   fun_data->B + fun_data->X_ind[ix]*k, 1) );
    }
    *f = reg_term - lsum * fun_data->w_mult;
}

void calc_grad_single(real_t a_row[], int k_int, real_t grad[], void *data)
{
    fdata* fun_data = (fdata*) data;
    size_t k = (size_t)k_int;
    memcpy(grad, fun_data->Bsum, sizeof(real_t) * k);
    cblas_taxpy(k_int, 2. * fun_data->l2_reg, a_row, 1, grad, 1);
    for (size_t ix = 0; ix < fun_data->nnz_this; ix++)
    {
        cblas_taxpy(k_int, - fun_data->Xr[ix]
                             / cblas_tdot(k_int, a_row, 1,
                                          fun_data->B + fun_data->X_ind[ix]*k, 1),
                    fun_data->B + fun_data->X_ind[ix]*k, 1, grad, 1);
    }
}

void calc_grad_single_w(real_t a_row[], int k_int, real_t grad[], void *data)
{
    fdata* fun_data = (fdata*) data;
    size_t k = (size_t)k_int;
    memset(grad, 0, k*sizeof(real_t));
    for (size_t ix = 0; ix < fun_data->nnz_this; ix++)
    {
        cblas_taxpy(k_int, - fun_data->Xr[ix]
                             / cblas_tdot(k_int, a_row, 1,
                                          fun_data->B + fun_data->X_ind[ix]*k, 1),
                    fun_data->B + fun_data->X_ind[ix]*k, 1, grad, 1);
    }
    cblas_tscal(k_int, fun_data->w_mult, grad, 1);
    cblas_taxpy(k_int, 1., fun_data->Bsum, 1, grad, 1);
    cblas_taxpy(k_int, 2. * fun_data->l2_reg, a_row, 1, grad, 1);
}

int calc_fun_and_grad
(
    real_t *restrict a_row,
    real_t *restrict f,
    real_t *restrict grad,
    void *data
)
{
    fdata *fun_data = (fdata*)data;
    int k_int = fun_data->k;
    size_t k = (size_t)k_int;

    real_t pred;
    real_t lsum = 0;
    memset(grad, 0, k*sizeof(real_t));
    for (size_t ix = 0; ix < fun_data->nnz_this; ix++)
    {
        pred = cblas_tdot(k_int, a_row, 1, fun_data->B + fun_data->X_ind[ix]*k, 1);
        cblas_taxpy(k_int, - fun_data->Xr[ix] / pred,
                    fun_data->B + fun_data->X_ind[ix]*k, 1, grad, 1);
        lsum += fun_data->Xr[ix] * log(pred);
    }

    if (fun_data->w_mult != 1.)
        cblas_tscal(k_int, fun_data->w_mult, grad, 1);
    cblas_taxpy(k_int, 1., fun_data->Bsum, 1, grad, 1);
    real_t reg_term = cblas_tdot(k_int, fun_data->Bsum, 1, a_row, 1);
    cblas_taxpy(k_int, 2. * fun_data->l2_reg, a_row, 1, grad, 1);

    *f = reg_term - lsum * fun_data->w_mult;
    return 0;
}

void cg_iteration
(
    real_t *A, real_t *B,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k, bool limit_step,
    real_t *Bsum, real_t l2_reg, real_t w_mult, size_t maxupd,
    real_t *buffer_arr, real_t *Bsum_w, int nthreads
)
{
    int k_int = (int) k;

    fdata data = { B, Bsum, NULL, NULL, 0, l2_reg, w_mult, k_int };
    real_t fun_val;
    size_t niter;
    size_t nfeval;
    grad_eval *grad_fun = (w_mult == 1.)? calc_grad_single : calc_grad_single_w;

    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ia;
    #endif

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            private(fun_val, niter, nfeval) firstprivate(data) \
            shared(dimA, Xr, Xr_indptr, Xr_indices, A, k, k_int, grad_fun)
    for (size_t_for ia = 0; ia < dimA; ia++)
    {
        if (should_stop_procedure)
            continue;

        data.Xr = Xr + Xr_indptr[ia];
        data.X_ind = Xr_indices + Xr_indptr[ia];
        data.nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];

        if (data.nnz_this == 0) {
            memset(A + ia*k, 0, k*sizeof(real_t));
            continue;
        }

        if (w_mult != 1.) data.Bsum = Bsum_w + ia*k;

        minimize_nonneg_cg(
            A + ia*k, k_int, &fun_val,
            calc_fun_single, grad_fun, NULL, (void*) &data,
            1e-2, 150, maxupd, &niter, &nfeval,
            0.25, 0.01, 20, limit_step,
            buffer_arr + 5*k*omp_get_thread_num(), 1, 0);
    }
}

void tncg_iteration
(
    real_t *A, real_t *B, bool reuse_prev,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    real_t *Bsum, real_t l2_reg, real_t w_mult, int maxupd,
    real_t *buffer_arr, int *buffer_int,
    real_t *restrict buffer_unchanged, bool *has_converged,
    real_t *zeros_tncg, real_t *inf_tncg,
    real_t *Bsum_w, int nthreads
)
{
    int k_int = (int) k;

    fdata data = { B, Bsum, NULL, NULL, 0, l2_reg, w_mult, k_int };
    real_t fun_val = 0;
    int niter = 0;
    int nfeval = 0;
    int maxCGit = (int) fmax(1., fmin(50., (real_t)k/2.));

    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ia;
    #endif

    *has_converged = false;
    size_t n_unchanged = 0;
    real_t *prev_values;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            firstprivate(data) private(niter, nfeval, fun_val, prev_values) \
            shared(A, dimA, Bsum_w, k, k_int, zeros_tncg, inf_tncg, \
                   buffer_arr, buffer_int, Xr, Xr_indices, Xr_indptr, \
                   maxupd, w_mult) \
            reduction(+:n_unchanged)
    for (size_t_for ia = 0; ia < dimA; ia++)
    {
        if (should_stop_procedure)
            continue;
        
        data.Xr = Xr + Xr_indptr[ia];
        data.X_ind = Xr_indices + Xr_indptr[ia];
        data.nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];

        if (data.nnz_this == 0) {
            memset(A + ia*k, 0, k*sizeof(real_t));
            continue;
        }

        if (w_mult != 1.) data.Bsum = Bsum_w + ia*k;

        if (buffer_unchanged != NULL) {
            prev_values = buffer_unchanged + k*(size_t)omp_get_thread_num();
            memcpy(prev_values, A + ia*k, k*sizeof(real_t));
        }

        if (!reuse_prev)
            for (size_t ix = 0; ix < k; ix++)
                A[ia*k + ix] = 1e-3;
        
        tnc(k_int, A + ia*k, &fun_val,
            buffer_arr + (size_t)omp_get_thread_num()*(size_t)22*k + (size_t)21*k,
            calc_fun_and_grad, (void*) &data,
            zeros_tncg, inf_tncg, NULL, NULL,
            0, maxCGit, maxupd, 0.25, 10.,
            0., 0., 1e-4, -1., -1.,
            1.3, &nfeval, &niter,
            buffer_arr + (size_t)omp_get_thread_num()*(size_t)22*k,
            buffer_int + (size_t)omp_get_thread_num()*k);

        if (buffer_unchanged != NULL) {
            cblas_taxpy(k_int, -1., A + ia*k, 1, prev_values, 1);
            n_unchanged += cblas_tdot(k_int, prev_values, 1, prev_values, 1) <= 1e-4;
        }
    }

    /* TODO: better keep an entry-by-entry array of whether they've changed,
       then examine skipping them individually instead. */
    if (buffer_unchanged != NULL) {
        *has_converged = ((double)n_unchanged / (double)dimA) >= .95;
    }
}


/* Main function for Proximal Gradient and Conjugate Gradient solvers
    A                         : Pointer to the already-initialized A matrix
                                (user factors)
    Xr, Xr_indptr, Xr_indices : Pointers to the X matrix in row-sparse format
    B                         : Pointer to the already-initialized B matrix
                                (item factors)
    Xc, Xc_indptr, Xc_indices : Pointers to the X matrix in column-sparse format
    dimA                      : Number of rows in the A matrix
    dimB                      : Number of rows in the B matrix
    k                         : Dimensionality for the factorizing matrices
                                (number of columns of A and B matrices)
    l2_reg                    : Regularization pameter for the L2 norm of the A and B matrices
    l1_reg                    : Regularization pameter for the L1 norm of the A and B matrices
    w_mult                    : Weight multiplier for the positive entries in X
    step_size                 : Initial step size for PGD updates
                                (will be decreased by 1/2 every iteration - ignored for CG)
    method                    : Which optimization method to use (tncg, cg, pg).
    limit_step                : Whether to limit CG step sizes to zero-out one variable per step
    numiter                   : Number of iterations for which to run the procedure
    maxupd                    : Number of updates to the same vector per iteration
    early_stop                : Whether to stop early if the values do not change much after an iteration (TNCG)
    reuse_prev                : Whether to re-use previous values as starting point (TNCG)
    handle_interrupt          : Whether to stop gracefully after a SIGINT, returning code 2 instead.
    nthreads                  : Number of threads to use
Matrices A and B are optimized in-place,
and are assumed to be in row-major order.
Returns 0 if it succeeds, 1 if it runs out of memory, 2 if it gets interrupted.
*/
int run_poismf(
    real_t *restrict A, real_t *restrict Xr, sparse_ix *restrict Xr_indptr, sparse_ix *restrict Xr_indices,
    real_t *restrict B, real_t *restrict Xc, sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_indices,
    const size_t dimA, const size_t dimB, const size_t k,
    const real_t l2_reg, const real_t l1_reg, const real_t w_mult, real_t step_size,
    const Method method, const bool limit_step, const size_t numiter, const size_t maxupd,
    const bool early_stop, const bool reuse_prev,
    const bool handle_interrupt, const int nthreads)
{
    sig_t_ old_interrupt_handle = NULL;
    bool has_lock_on_handle = false;
    #pragma omp critical
    {
        if (!handle_is_locked)
        {
            handle_is_locked = true;
            has_lock_on_handle = true;
            should_stop_procedure = false;
            old_interrupt_handle = signal(SIGINT, set_interrup_global_variable);
        }
    }

    real_t *cnst_sum = (real_t*) malloc(sizeof(real_t) * k);
    real_t cnst_div;
    int k_int = (int) k;
    real_t neg_step_sz = -step_size;
    size_t size_buffer = 1;
    switch(method) {
        case pg:   {size_buffer = 1;  break;}
        case cg:   {size_buffer = 5;  break;}
        case tncg: {size_buffer = 22; break;}
    }
    size_buffer *= (k * (size_t)nthreads);
    real_t *buffer_arr = (real_t*) malloc(sizeof(real_t) * size_buffer);
    real_t *Bsum_w = NULL;
    int *buffer_int = NULL;
    real_t *zeros_tncg = NULL;
    real_t *inf_tncg = NULL;
    real_t *buffer_unchanged = NULL;
    bool stopped_earlyA = false, stopped_earlyB = false;
    int ret_code = 0;


    if (w_mult != 1.) {
        Bsum_w = (real_t*)malloc(sizeof(real_t) * k * ((dimA > dimB)? dimA : dimB));
        if (Bsum_w == NULL) goto throw_oom;
    }

    if (method == tncg) {
        buffer_int = (int*)malloc(sizeof(int) * k *(size_t)nthreads);
        zeros_tncg = (real_t*)calloc(sizeof(real_t), k);
        inf_tncg = (real_t*)malloc(sizeof(real_t) * k);
        if (buffer_int == NULL || zeros_tncg == NULL || inf_tncg == NULL)
            goto throw_oom;
        if (early_stop) {
            buffer_unchanged = (real_t*)malloc((size_t)nthreads*k*sizeof(real_t));
            if (buffer_unchanged == NULL)
                goto throw_oom;
        }
        for (size_t ix = 0; ix < k; ix++)
            inf_tncg[ix] = HUGE_VAL;
    }

    if (buffer_arr == NULL || cnst_sum == NULL)
    {
        throw_oom:
            fprintf(stderr, "Error: out of memory.\n");
            ret_code = 1;
            goto cleanup;
    }

    for (size_t fulliter = 0; fulliter < numiter; fulliter++){

        if (should_stop_procedure) goto cleanup;

        /* Constants to use later */
        cnst_div = 1. / (1. + 2. * l2_reg * step_size);
        sum_by_cols(cnst_sum, A, dimA, k);
        if (l1_reg > 0.)
            for (size_t kk = 0; kk < k; kk++) cnst_sum[kk] += l1_reg;
        if (w_mult != 1.)
            adjustment_Bsum(A, cnst_sum, Bsum_w,
                            Xc_indices, Xc_indptr, dimB, k,
                            w_mult, nthreads);

        switch(method) {
            case pg:
            {
                if (w_mult == 1.)
                    cblas_tscal(k_int, neg_step_sz, cnst_sum, 1);
                else
                    dscal_large(dimB*k, neg_step_sz, Bsum_w);
                pg_iteration(B, A, Xc, Xc_indptr, Xc_indices,
                             dimB, k, cnst_div, cnst_sum, Bsum_w, step_size,
                             w_mult, maxupd, buffer_arr, nthreads);

                /* Decrease step size after taking PGD steps in both matrices */
                step_size *= 0.5;
                neg_step_sz = -step_size;
                break;
            }

            case cg:
            {
                cg_iteration(B, A, Xc, Xc_indptr, Xc_indices,
                             dimB, k, limit_step, cnst_sum,
                             l2_reg, w_mult, maxupd,
                             buffer_arr, Bsum_w, nthreads);
                break;
            }

            case tncg:
            {
                if (!stopped_earlyB)
                tncg_iteration(B, A, reuse_prev, Xc, Xc_indptr, Xc_indices,
                               dimB, k, cnst_sum, l2_reg, w_mult, maxupd,
                               buffer_arr, buffer_int,
                               buffer_unchanged, &stopped_earlyB,
                               zeros_tncg, inf_tncg,
                               Bsum_w, nthreads);
                break;
            }
        }

        if (should_stop_procedure) goto cleanup;

         /* Same procedure repeated for the A matrix */
        sum_by_cols(cnst_sum, B, dimB, k);
        if (l1_reg > 0.)
            for (size_t kk = 0; kk < k; kk++) cnst_sum[kk] += l1_reg;
        if (w_mult != 1.)
            adjustment_Bsum(B, cnst_sum, Bsum_w,
                            Xr_indices, Xr_indptr, dimA, k,
                            w_mult, nthreads);

        switch (method) {
            case pg:
            {
                if (w_mult == 1.)
                    cblas_tscal(k_int, neg_step_sz, cnst_sum, 1);
                else
                    dscal_large(dimA*k, neg_step_sz, Bsum_w);
                cblas_tscal(k_int, neg_step_sz, cnst_sum, 1);
                pg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                             dimA, k, cnst_div, cnst_sum, Bsum_w, step_size,
                             w_mult, maxupd, buffer_arr, nthreads);
                break;
            }

            case cg:
            {
                cg_iteration(A, B, Xr, Xr_indptr, Xr_indices,
                             dimA, k, limit_step, cnst_sum,
                             l2_reg, w_mult, maxupd,
                             buffer_arr, Bsum_w, nthreads);
                break;
            }

            case tncg:
            {
                if (!stopped_earlyA)
                tncg_iteration(A, B, reuse_prev, Xr, Xr_indptr, Xr_indices,
                               dimA, k, cnst_sum, l2_reg, w_mult, maxupd,
                               buffer_arr, buffer_int,
                               buffer_unchanged, &stopped_earlyA,
                               zeros_tncg, inf_tncg,
                               Bsum_w, nthreads);
                break;
            }
        }

        if (stopped_earlyA && stopped_earlyB)
            break;
    }

    cleanup:
        free(cnst_sum);
        free(buffer_arr);
        free(buffer_int);
        free(Bsum_w);
        free(buffer_unchanged);
        free(zeros_tncg);
        free(inf_tncg);
        #pragma omp critical
        {
            bool should_stop_procedure_local = should_stop_procedure;
            if (should_stop_procedure_local && ret_code != 1)
                ret_code = 2;
            if (has_lock_on_handle) {
                signal(SIGINT, old_interrupt_handle);
                handle_is_locked = false;
                should_stop_procedure = false;
            }
            if (should_stop_procedure_local && !handle_interrupt)
                raise(SIGINT);
        }
    return ret_code;
}

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

#ifdef __cplusplus
extern "C" {
#endif

/* Aliasing for compiler optimizations */
#ifdef __cplusplus
    #if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER)
        #define restrict __restrict
    #else
        #define restrict 
    #endif
#elif defined(_MSC_VER)
    #define restrict __restrict
#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
    #define restrict 
#endif
/* Note: MSVC is a special boy which does not allow 'restrict' in C mode,
   so don't move this piece of code down with the others, otherwise the
   function prototypes will not compile */


#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <limits.h>
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() (0)
#endif
#ifndef _FOR_R
    #define sparse_ix size_t
    #include <stdio.h>
#else
    #include <Rinternals.h>
    #include <R.h>
    #include <R_ext/Rdynload.h>
    #include <R_ext/Print.h>
    #define fprintf(f, ...) REprintf(__VA_ARGS__)
    #define sparse_ix int
    #undef USE_FLOAT
#endif
#include <signal.h>
typedef void (*sig_t_)(int);

#ifndef USE_FLOAT
    #ifndef real_t
        #define real_t double
    #endif
    #define cblas_tdot cblas_ddot
    #define cblas_taxpy cblas_daxpy
    #define cblas_tscal cblas_dscal
    #define cblas_tnrm2 cblas_dnrm2
    #define cblas_tgemv cblas_dgemv
#else
    #ifndef real_t
        #define real_t float
    #endif
    #define cblas_tdot cblas_sdot
    #define cblas_taxpy cblas_saxpy
    #define cblas_tscal cblas_sscal
    #define cblas_tnrm2 cblas_snrm2
    #define cblas_tgemv cblas_sgemv
#endif

/* https://www.github.com/david-cortes/nonneg_cg */
typedef void fun_eval(real_t x[], int n, real_t *f, void *data);
typedef void grad_eval(real_t x[], int n, real_t grad[], void *data);
typedef void callback(real_t x[], int n, real_t f, size_t iter, void *data);
int minimize_nonneg_cg(real_t x[], int n, real_t *fun_val,
                       fun_eval *obj_fun, grad_eval *grad_fun, callback *cb, void *data,
                       real_t tol, size_t maxnfeval, size_t maxiter, size_t *niter, size_t *nfeval,
                       real_t decr_lnsrch, real_t lnsrch_const, size_t max_ls,
                       bool limit_step, real_t *buffer_arr, int nthreads, int verbose);
/* Data struct to pass to nonneg_cg */
typedef struct fdata {
    real_t *B;
    real_t *Bsum;
    real_t *Xr;
    sparse_ix *X_ind;
    sparse_ix nnz_this;
    real_t l2_reg;
    real_t w_mult;
    int k;
} fdata;

/* TNC */
#include "tnc.h"

/* BLAS functions */
#ifndef CBLAS_H
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef CBLAS_ORDER CBLAS_LAYOUT;
real_t cblas_tdot(const int n, const real_t *x, const int incx, const real_t *y, const int incy);
void cblas_taxpy(const int n, const real_t alpha, const real_t *x, const int incx, real_t *y, const int incy);
void cblas_tscal(const int N, const real_t alpha, real_t *X, const int incX);
real_t cblas_tnrm2(const int n, const real_t *x, const int incx);
#ifndef _FOR_PYTHON
void cblas_tgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE TransA,  const int m, const int n,
     const real_t alpha, const real_t  *a, const int lda,  const real_t  *x, const int incx,  const real_t beta,  real_t  *y, const int incy);
#else /* <- Cython refuses to compile a cdef'd public type with enum arguments */
void cblas_tgemv(const int order,  const int TransA,  const int m, const int n,
     const real_t alpha, const real_t  *a, const int lda,  const real_t  *x, const int incx,  const real_t beta,  real_t  *y, const int incy);
#endif
#endif

/* Visual Studio as of 2019 is stuck with OpenMP 2.0 (released 2002),
   which doesn't support parallel loops with unsigned iterators,
   and doesn't support declaring a for-loop iterator in the loop itself. */
#ifdef _OPENMP
    #if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
        #define size_t_for 
    #else
        #define size_t_for size_t
    #endif
#else
    #define size_t_for size_t
#endif

/* Function prototypes */

/* poismf.c */
void dscal_large(size_t n, real_t alpha, real_t *restrict x);
void sum_by_cols(real_t *restrict out, real_t *restrict M, size_t nrow, size_t ncol);
void adjustment_Bsum
(
    real_t *restrict B,
    real_t *restrict Bsum,
    real_t *restrict Bsum_user,
    sparse_ix Xr_indices[],
    sparse_ix Xr_indptr[],
    size_t dimA, size_t k,
    real_t w_mult, int nthreads
);
void calc_grad_pgd(real_t *out, real_t *curr, real_t *F, real_t *X, sparse_ix *Xind, sparse_ix nnz_this, int k);
void pg_iteration
(
    real_t *A, real_t *B,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    real_t cnst_div, real_t *cnst_sum, real_t *Bsum_user,
    real_t step_size, real_t w_mult, size_t maxupd,
    real_t *buffer_arr, int nthreads
);
void calc_fun_single(real_t a_row[], int k_int, real_t *f, void *data);
void calc_grad_single(real_t a_row[], int k_int, real_t grad[], void *data);
void calc_grad_single_w(real_t a_row[], int k_int, real_t grad[], void *data);
int calc_fun_and_grad
(
    real_t *restrict a_row,
    real_t *restrict f,
    real_t *restrict grad,
    void *data
);
void cg_iteration
(
    real_t *A, real_t *B,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k, bool limit_step,
    real_t *Bsum, real_t l2_reg, real_t w_mult, size_t maxupd,
    real_t *buffer_arr, real_t *Bsum_w, int nthreads
);
void tncg_iteration
(
    real_t *A, real_t *B, bool reuse_prev,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    real_t *Bsum, real_t l2_reg, real_t w_mult, int maxupd,
    real_t *buffer_arr, int *buffer_int,
    real_t *buffer_unchanged, bool *has_converged,
    real_t *zeros_tncg, real_t *inf_tncg,
    real_t *Bsum_w, int nthreads
);
void set_interrup_global_variable(int s);

/* main function */
typedef enum Method {tncg = 1, cg = 2, pg = 3} Method;
int run_poismf(
    real_t *restrict A, real_t *restrict Xr, sparse_ix *restrict Xr_indptr, sparse_ix *restrict Xr_indices,
    real_t *restrict B, real_t *restrict Xc, sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_indices,
    const size_t dimA, const size_t dimB, const size_t k,
    const real_t l2_reg, const real_t l1_reg, const real_t w_mult, real_t step_size,
    const Method method, const bool limit_step, const size_t numiter, const size_t maxupd,
    const bool early_stop, const bool reuse_prev,
    const bool handle_interrupt, const int nthreads);

/* topN.c */
bool check_is_sorted(sparse_ix arr[], size_t n);
void qs_argpartition(sparse_ix arr[], real_t values[], size_t n, size_t k);
int cmp_size_t(const void *a, const void *b);
int cmp_argsort(const void *a, const void *b);
int topN
(
    real_t *restrict a_vec, real_t *restrict B, int k,
    sparse_ix *restrict include_ix, size_t n_include,
    sparse_ix *restrict exclude_ix, size_t n_exclude,
    sparse_ix *restrict outp_ix, real_t *restrict outp_score,
    size_t n_top, size_t n, int nthreads
);

/* llk_and_pred.c */
void predict_multiple
(
    real_t *restrict out,
    real_t *restrict A, real_t *restrict B,
    sparse_ix *ixA, sparse_ix *ixB,
    size_t n, int k,
    int nthreads
);
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
);
int factors_multiple
(
    real_t *A, real_t *B,
    real_t *Bsum, real_t *Amean,
    real_t *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    int k, size_t dimA,
    real_t l2_reg, real_t w_mult,
    real_t step_size, size_t niter, size_t maxupd,
    Method method, bool limit_step, bool reuse_mean,
    int nthreads
);
int factors_single
(
    real_t *restrict out, size_t k,
    real_t *restrict Amean, bool reuse_mean,
    real_t *restrict X, sparse_ix X_ind[], size_t nnz,
    real_t *restrict B, real_t *restrict Bsum,
    int maxupd, real_t l2_reg, real_t l1_new, real_t l1_old,
    real_t w_mult
);

#ifdef __cplusplus
}
#endif


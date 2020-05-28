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
#include <signal.h>
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
#endif

/* https://www.github.com/david-cortes/nonneg_cg */
typedef void fun_eval(double x[], int n, double *f, void *data);
typedef void grad_eval(double x[], int n, double grad[], void *data);
typedef void callback(double x[], int n, double f, size_t iter, void *data);
int minimize_nonneg_cg(double x[], int n, double *fun_val,
                       fun_eval *obj_fun, grad_eval *grad_fun, callback *cb, void *data,
                       double tol, size_t maxnfeval, size_t maxiter, size_t *niter, size_t *nfeval,
                       double decr_lnsrch, double lnsrch_const, size_t max_ls,
                       bool limit_step, double *buffer_arr, int nthreads, int verbose);
/* Data struct to pass to nonneg_cg */
typedef struct fdata {
    double *B;
    double *Bsum;
    double *Xr;
    sparse_ix *X_ind;
    sparse_ix nnz_this;
    double l2_reg;
    double w_mult;
    int k;
} fdata;

/* TNC */
#include "tnc.h"

/* BLAS functions */

#ifdef _FOR_PYTHON
    #include "findblas.h"  /* https://www.github.com/david-cortes/findblas */
#else
    #ifndef CBLAS_H
    typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
    typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
    typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
    typedef CBLAS_ORDER CBLAS_LAYOUT;
    double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
    void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
    void cblas_dscal(const int N, const double alpha, double *X, const int incX);
    double cblas_dnrm2(const int n, const double *x, const int incx);
    void cblas_dgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE TransA,  const int m, const int n,
         const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy);
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
void dscal_large(size_t n, double alpha, double *restrict x);
void sum_by_cols(double *restrict out, double *restrict M, size_t nrow, size_t ncol);
void adjustment_Bsum
(
    double *restrict B,
    double *restrict Bsum,
    double *restrict Bsum_user,
    sparse_ix Xr_indices[],
    sparse_ix Xr_indptr[],
    size_t dimA, size_t k,
    double w_mult, int nthreads
);
void calc_grad_pgd(double *out, double *curr, double *F, double *X, sparse_ix *Xind, sparse_ix nnz_this, int k);
void pg_iteration
(
    double *A, double *B,
    double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    double cnst_div, double *cnst_sum, double *Bsum_user,
    double step_size, double w_mult, size_t maxupd,
    double *buffer_arr, int nthreads
);
void calc_fun_single(double a_row[], int k_int, double *f, void *data);
void calc_grad_single(double a_row[], int k_int, double grad[], void *data);
void calc_grad_single_w(double a_row[], int k_int, double grad[], void *data);
int calc_fun_and_grad
(
    double *restrict a_row,
    double *restrict f,
    double *restrict grad,
    void *data
);
void cg_iteration
(
    double *A, double *B,
    double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k, bool limit_step,
    double *Bsum, double l2_reg, double w_mult, size_t maxupd,
    double *buffer_arr, double *Bsum_w, int nthreads
);
void tncg_iteration
(
    double *A, double *B,
    double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    size_t dimA, size_t k,
    double *Bsum, double l2_reg, double w_mult, int maxupd,
    double *buffer_arr, int *buffer_int,
    double *zeros_tncg, double *inf_tncg,
    double *Bsum_w, int nthreads
);
void set_interrup_global_variable(int s);

/* main function */
typedef enum Method {tncg = 1, cg = 2, pg = 3} Method;
int run_poismf(
    double *restrict A, double *restrict Xr, sparse_ix *restrict Xr_indptr, sparse_ix *restrict Xr_indices,
    double *restrict B, double *restrict Xc, sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_indices,
    const size_t dimA, const size_t dimB, const size_t k,
    const double l2_reg, const double l1_reg, const double w_mult, double step_size,
    const Method method, const bool limit_step, const size_t numiter, const size_t maxupd,
    const int nthreads);

/* topN.c */
bool check_is_sorted(sparse_ix arr[], size_t n);
void qs_argpartition(sparse_ix arr[], double values[], size_t n, size_t k);
int cmp_size_t(const void *a, const void *b);
int cmp_argsort(const void *a, const void *b);
int topN
(
    double *restrict a_vec, double *restrict B, int k,
    sparse_ix *restrict include_ix, size_t n_include,
    sparse_ix *restrict exclude_ix, size_t n_exclude,
    sparse_ix *restrict outp_ix, double *restrict outp_score,
    size_t n_top, size_t n, int nthreads
);

/* llk_and_pred.c */
void predict_multiple
(
    double *restrict out,
    double *restrict A, double *restrict B,
    sparse_ix *ixA, sparse_ix *ixB,
    size_t n, int k,
    int nthreads
);
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
);
int factors_multiple
(
    double *A, double *B, double *A_old, double *Bsum,
    double *Xr, sparse_ix *Xr_indptr, sparse_ix *Xr_indices,
    int k, size_t dimA,
    double l2_reg, double w_mult,
    double step_size, size_t niter, size_t maxupd,
    Method method, bool limit_step,
    int nthreads
);
int factors_single
(
    double *restrict out, size_t k,
    double *restrict A_old, size_t dimA,
    double *restrict X, sparse_ix X_ind[], size_t nnz,
    double *restrict B, double *restrict Bsum,
    int maxupd, double l2_reg, double l1_new, double l1_old,
    double w_mult
);

#ifdef __cplusplus
}
#endif


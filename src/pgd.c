 /*
	Poisson Factorization for sparse matrices

	Based on alternating proximal gradient iteration.
	Variables must be initialized from outside the main function provided here.
	Writen for C99 standard.

	BSD 2-Clause License

	Copyright (c) 2019, David Cortes
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

/* Aliasing for compiler optimizations */
#ifndef restrict
	#ifdef __restrict
		#define restrict __restrict
	#elif defined(__restrict__)
		#define restrict __restrict__
	#else
		#define restrict
	#endif
#endif
/* Note: MSVC is a special boy which does not allow 'restrict' in C mode,
   so don't move this piece of code down with the others, otherwise the
   function prototypes will not compile */


#include <stdbool.h>
#include <string.h>
#include <math.h>
#ifndef _FOR_R
	#include <stdio.h>
#endif
#include <stddef.h>
#ifndef _FOR_R
	/* https://www.github.com/david-cortes/nonneg_cg */
	typedef void fun_eval(double x[], int n, double *f, void *data);
	typedef void grad_eval(double x[], int n, double grad[], void *data);
	typedef void callback(double x[], int n, double f, size_t iter, void *data);
	int minimize_nonneg_cg(double x[restrict], int n, double *fun_val,
						   fun_eval *obj_fun, grad_eval *grad_fun, callback *cb, void *data,
						   double tol, size_t maxnfeval, size_t maxiter, size_t *niter, size_t *nfeval,
						   double decr_lnsrch, double lnsrch_const, size_t max_ls,
						   int extra_nonneg_tol, double *buffer_arr, int nthreads, int verbose);
#else
	#include <Rinternals.h>
	#include <R.h>
	#include <R_ext/Rdynload.h>
	#include <R_ext/Print.h>
	#define fprintf(f, message) REprintf(message)
#endif

/* BLAS functions */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _FOR_PYTHON
	#include "findblas.h"  /* https://www.github.com/david-cortes/findblas */
#elif defined(_FOR_R)
	#include <R_ext/BLAS.h>
	double cblas_ddot(int n, double *x, int incx, double *y, int incy) { return ddot_(&n, x, &incx, y, &incy); }
	void cblas_daxpy(int n, double a, double *x, int incx, double *y, int incy) { daxpy_(&n, &a, x, &incx, y, &incy); }
	void cblas_dscal(int n, double alpha, double *x, int incx) { dscal_(&n, &alpha, x, &incx); }
#else
	double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
	void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
	void cblas_dscal(const int N, const double alpha, double *X, const int incX);

	#ifndef cblas_ddot
		double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) {
			double out = 0;
			for (int i = 0; i < n; i++) { out += x[i] * y[i]; }
			return out;
		}
		void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) { for (int i = 0; i < n; i++) { y[i] += alpha * x[i]; } }
		void cblas_dscal(const int N, const double alpha, double *X, const int incX) { for (int i = 0; i < N; i++) { X[i] *= alpha; } }
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

/* Check if variable-length arrays are supported - this is used for parallelizing
   sums of columns
   https://cboard.cprogramming.com/c-programming/176320-vla-pointers-vla-checking-isoc90-isoc11.html
*/
#if __STDC_VERSION__ >= 201100L
	#ifndef _STDC_NO_VLA_
		#define HAS_VLA
	#endif
#elif __STDC_VERSION__ >= 199900L
	#define HAS_VLA
#endif

/* Helper functions */
#define nonneg(x) ((x) > 0)? (x) : 0

void sum_by_cols(double *restrict out, double *restrict M, size_t nrow, size_t ncol, int ncores)
{
	memset(out, 0, sizeof(double) * ncol);

	#if !defined(_MSC_VER) && defined(HAS_VLA) && (_OPENMP > 200801) && !defined(_FOR_R) && !defined(_WIN32) && !defined(_WIN64)
	/* DAMN YOU MS, WHY WON'T YOU SUPPORT SUCH BASIC FUNCTIONALITY!!! */
	/* From CRAN: this also fails in (oracle) solaris */
	#pragma omp parallel for if(ncol <= 100) schedule(static, nrow/ncores) num_threads(ncores) firstprivate(nrow, ncol, M) reduction(+:out[:ncol])
	#endif
	for (size_t row = 0; row < nrow; row++){
		for (size_t col = 0; col < ncol; col++){
			out[col] += M[row*ncol + col];
		}
	}
}

/*	OpenMP parallel buffer arrays.
	Should ideally be rather used as a proper array[k], and listed in 'omp private(array)',
	but for compatibility with MS Visual Studio which supports neiter C99 nor OpenMP>=3.0,
	it was coded like this, with global variables. */
double *buffer_arr;
#pragma omp threadprivate(buffer_arr)

/* Functions for Proximal Gradient */
void calc_grad_pgd(double *out, double *curr, double *F, double *X, size_t *Xind, size_t nnz_this, int k)
{
	memset(out, 0, sizeof(double) * k);
	for (size_t i = 0; i < nnz_this; i++){
		cblas_daxpy(k, X[i] / cblas_ddot(k, F + Xind[i] * k, 1, curr, 1), F + Xind[i] * k, 1, out, 1);
	}
}

/*	This function is written having in mind the A matrix being optimized, with the B matrix being fixed, and the data passed in row-sparse format.
	For optimizing B, swap any mention of A and B, and pass the data in column-sparse format */
void pgd_iteration(double *A, double *B, double *Xr, size_t *Xr_indptr, size_t *Xr_indices, size_t dimA, size_t k,
	double cnst_div, double *cnst_sum, double step_size, size_t npass, int ncores)
{
	int k_int = (int) k;
	size_t nnz_this;

	#ifdef _OPENMP
		#if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
			long ia;
		#endif
	#endif

	#pragma omp parallel for schedule(dynamic) num_threads(ncores) shared(A) private(nnz_this) firstprivate(B, k, k_int, cnst_sum, cnst_div, npass, Xr, Xr_indptr, Xr_indices)
	for (size_t_for ia = 0; ia < dimA; ia++)
	{

		nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];
		for (size_t p = 0; p < npass; p++)
		{
			calc_grad_pgd(buffer_arr, A + ia*k, B, Xr + Xr_indptr[ia], Xr_indices + Xr_indptr[ia], nnz_this, k_int);
			cblas_daxpy(k_int, step_size, buffer_arr, 1, A + ia*k, 1);

			cblas_daxpy(k_int, 1, cnst_sum, 1, A + ia*k, 1);
			cblas_dscal(k_int, cnst_div, A + ia*k, 1);
			for (size_t i = 0; i < k; i++) {A[ia*k + i] = nonneg(A[ia*k + i]);}
		}

	}
}

#ifndef _FOR_R
/* Functions and structs for Conjugate Gradient - these are used with package nonneg_cg */
typedef struct fdata {
	double *F;
	double *Fsum;
	double *X;
	size_t *X_ind;
	size_t nnz_this;
	double l2_reg;
} fdata;

void calc_fun_single(double x[], int n, double *f, void *data)
{
	fdata* fun_data = (fdata*) data;
	double out = cblas_ddot(n, fun_data->Fsum, 1, x, 1);
	double norm_sq = cblas_ddot(n, x, 1, x, 1);
	out += fun_data->l2_reg * norm_sq;
	for (size_t i = 0; i < fun_data->nnz_this; i++)
	{
		out -= fun_data->X[i] * log( cblas_ddot(n, x, 1, fun_data->F + fun_data->X_ind[i] * n, 1) );
	}
	*f = out;
}

void calc_grad_single(double x[], int n, double grad[], void *data)
{
	fdata* fun_data = (fdata*) data;
	memcpy(grad, fun_data->Fsum, sizeof(double) * n);
	cblas_daxpy(n, 2 * n * fun_data->l2_reg, x, 1, grad, 1);
	for (size_t i = 0; i < fun_data->nnz_this; i++)
	{
		cblas_daxpy(n, - fun_data->X[i] / cblas_ddot(n, x, 1, fun_data->F + fun_data->X_ind[i] * n, 1),
					fun_data->F + fun_data->X_ind[i] * n, 1, grad, 1);
	}
}

void optimize_cg_single(double curr[], double X[], size_t X_ind[], size_t nnz_this, double F[], double Fsum[], int k, double l2_reg)
{

	fdata data = { F, Fsum, X, X_ind, nnz_this, l2_reg };
	double fun_val;
	size_t niter;
	size_t nfeval;

	minimize_nonneg_cg(
		curr, k, &fun_val,
		calc_fun_single, calc_grad_single, NULL, (void*) &data,
		1e-1, 200, 100, &niter, &nfeval,
		0.25, 0.01, 20,
		1, NULL, 1, 0);
}

void cg_iteration(double *A, double *B, double *Xr, size_t *Xr_indptr, size_t *Xr_indices, size_t dimA, size_t k,
	double *Bsum, size_t npass, double l2_reg, int ncores)
{

	int k_int = (int) k;

	fdata data = { B, Bsum, NULL, NULL, 0, l2_reg };
	double fun_val;
	size_t niter;
	size_t nfeval;

	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
	long ia;
	#endif

	#pragma omp parallel for schedule(dynamic) num_threads(ncores) private(fun_val, niter, nfeval) firstprivate(data, dimA, Xr, Xr_indptr, Xr_indices, npass, A, k, k_int)
	for (size_t_for ia = 0; ia < dimA; ia++)
	{
		data.X = Xr + Xr_indptr[ia];
		data.X_ind = Xr_indices + Xr_indptr[ia];
		data.nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];

		minimize_nonneg_cg(
			A + ia*k, k_int, &fun_val,
			calc_fun_single, calc_grad_single, NULL, (void*) &data,
			1e-5, 150, 50, &niter, &nfeval,
			0.25, 0.01, 25,
			1, buffer_arr, 1, 0);
	}
}
#endif


/* Main function for Proximal Gradient and Conjugate Gradient solvers
	A                           : Pointer to the already-initialized A matrix (user-factor)
	Xr, Xr_indptr, Xr_indices   : Pointers to the X matrix in row-sparse format
	B                           : Pointer to the already-initialized B matrix (item-factor)
	Xc, Xc_indptr, Xc_indices   : Pointers to the X matrix in column-sparse format
	dimA                        : Number of rows in the A matrix
	dimB                        : Number of rows in the B matrix
	k                           : Dimensionality for the factorizing matrices (number of columns of A and B matrices)
	l2_reg                      : Regularization pameter for the L2 norm of the A and B matrices
	l1_reg                      : Regularization pameter for the L1 norm of the A and B matrices
	use_cg                      : Whether to use a Conjugate-Gradient solver instead of Proximal-Gradient.
	step_size                   : Initial step size for PGD updates (will be decreased by 1/2 every iteration - ignored for CG)
	numiter                     : Number of iterations for which to run the procedure
	npass                       : Number of updates to the same matrix per iteration
	ncores                      : Number of threads to use
Matrices A and B are optimized in-place.
Function does not have a return value.
*/
void run_poismf(
	double *restrict A, double *restrict Xr, size_t *restrict Xr_indptr, size_t *restrict Xr_indices,
	double *restrict B, double *restrict Xc, size_t *restrict Xc_indptr, size_t *restrict Xc_indices,
	const size_t dimA, const size_t dimB, const size_t k,
	const double l2_reg, const double l1_reg, const int use_cg, double step_size,
	const size_t numiter, const size_t npass, const int ncores)
{

	double *cnst_sum = (double*) malloc(sizeof(double) * k);
	double cnst_div;
	int k_int = (int) k;
	double neg_step_sz = -step_size;
	bool buffer_alloc_error = false;
	#pragma omp parallel firstprivate(use_cg)
	{
		if (use_cg) {
			buffer_arr = (double*) malloc(sizeof(double) * k * 4);
		} else {
			buffer_arr = (double*) malloc(sizeof(double) * k);
		}
		if (buffer_arr == NULL) {
			buffer_alloc_error = true;
		}
	}

	#pragma omp barrier
	if (buffer_alloc_error || cnst_sum == NULL) {
		fprintf(stderr, "Error: Could not allocate memory for the procedure.\n");
		goto cleanup;
	}

	/* Functions are already well parallelized by rows/columns, so BLAS should ideally run single-threaded */
	#if defined(_MKL_H_)
		mkl_set_num_threads_local(1);
	#elif defined(CBLAS_H)
		openblas_set_num_threads(1);
	#endif

	for (size_t fulliter = 0; fulliter < numiter; fulliter++){

		/* Constants to use later */
		cnst_div = 1 / (1 + 2 * l2_reg * step_size);
		sum_by_cols(cnst_sum, B, dimB, k, ncores);
		if (l1_reg > 0) { for (size_t kk = 0; kk < k; kk++) { cnst_sum[kk] += l1_reg; } }

		#ifndef _FOR_R
		if (use_cg) {
			cg_iteration(A, B, Xr, Xr_indptr, Xr_indices, dimA, k, cnst_sum, npass, l2_reg, ncores);
		} else {
		#endif
			cblas_dscal(k_int, neg_step_sz, cnst_sum, 1);
			pgd_iteration(A, B, Xr, Xr_indptr, Xr_indices, dimA, k, cnst_div, cnst_sum, step_size, npass, ncores);
		#ifndef _FOR_R
		}
		#endif


		/* Same procedure repeated for the B matrix */
		sum_by_cols(cnst_sum, A, dimA, k, ncores);
		if (l1_reg > 0) { for (size_t kk = 0; kk < k; kk++) { cnst_sum[kk] += l1_reg; } }

		#ifndef _FOR_R
		if (use_cg) {
			cg_iteration(B, A, Xr, Xc_indptr, Xc_indices, dimB, k, cnst_sum, npass, l2_reg, ncores);
		} else {
		#endif
			cblas_dscal(k_int, neg_step_sz, cnst_sum, 1);
			pgd_iteration(B, A, Xc, Xc_indptr, Xc_indices, dimB, k, cnst_div, cnst_sum, step_size, npass, ncores);

			/* Decrease step size after taking PGD steps in both matrices */
			step_size *= 0.5;
			neg_step_sz = -step_size;
		#ifndef _FOR_R
		}
		#endif

	}

	cleanup:
		free(cnst_sum);
		#pragma omp parallel
		{
			free(buffer_arr);
		}
}


#ifdef _FOR_PYTHON
/* Generic helper function that predicts multiple combinations of users and items from already-fit A and B matrices */
void predict_multiple(double *out, double *A, double *B, size_t *ix_u, size_t *ix_i, size_t n, int k, int nthreads)
{
	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
	long n_szt = (long) n;
	long i;
	#else
	size_t n_szt = n;
	#endif

	size_t k_szt = (size_t) k;
	for (size_t_for i = 0; i < n_szt; i++) {
		out[i] = cblas_ddot(k, A + ix_u[i] * k_szt, 1, B + ix_i[i] * k_szt, 1);
	}
}

#endif

#ifdef __cplusplus
}
#endif

 /* Poisson Factorization for sparse matrices

 Based on alternating proximal gradient iteration.
 Variables must be initialized from outside the main function provided here.
 Writen for C99 standard.

 Copyright David Cortes 2018 */

#include <string.h>
#include "findblas.h" /* https://github.com/david-cortes/findblas */

/* Aliasing for compiler optimizations */
#ifndef restrict
	#ifdef __restrict
		#define restrict __restrict
	#else
		#define restrict
	#endif
#endif

/* Visual Studio as of 2018 is stuck with OpenMP 2.0 (released 2002),
   which doesn't support parallel loops with unsigned iterators,
   and doesn't support declaring a for-loop iterator in the loop itself.
   As the code is wrapped in Cython and Cython does not support typdefs conditional on compiler,
   this will map size_t to long on Windows regardless of compiler.
   Can be safely removed if not compiling with MSVC. */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
	#define size_t long
#else
	#include <stddef.h>
#endif
#ifdef _OPENMP
	#if _OPENMP < 20080101 /* OpenMP < 3.0 */
		#define size_t_for size_t
	#else
		#define size_t_for
	#endif
#else
	#define size_t_for size_t
#endif

/* Helper functions */
#define nonneg(x) (x >= 0)? x : 0

void sum_by_cols(double *restrict out, double *restrict M, size_t nrow, size_t ncol, int ncores)
{
	memset(out, 0, sizeof(double) * ncol);

	#if !defined(_MSC_VER) && _OPENMP>20080101 /* OpenMP >= 3.0 */
	/* DAMN YOU MS, WHY WON'T YOU SUPPORT SUCH BASIC FUNCTIONALITY!!! */
	#pragma omp parallel for schedule(static) num_threads(ncores) firstprivate(nrow, ncol, M) reduction(+:out[:ncol])
	#endif
	for (size_t row = 0; row < nrow; row++){
		for (size_t col = 0; col < ncol; col++){
			out[col] += M[row*ncol + col];
		}
	}
}

void calc_grad(double *out, double *curr, double *F, double *X, size_t *Xind, size_t nnz_this, int k)
{
	memset(out, 0, sizeof(double) * k);
	for (size_t i = 0; i < nnz_this; i++){
		cblas_daxpy(k, X[i] / cblas_ddot(k, F + Xind[i]*k, 1, curr, 1), F + Xind[i]*k, 1, out, 1);
	}
}

/*	OpenMP parallel buffer arrays.
	Should ideally be rather used as a proper array[k], and listed in 'omp private(array)',
	but for compatibility with MS Visual Studio which supports neiter C99 nor OpenMP>=3.0,
	it was coded like this, with global variables. */
double *buffer_arr;
#pragma omp threadprivate(buffer_arr)


/*	This function is written having in mind the A matrix being optimized, with the B matrix being fixed, and the data passed in row-sparse format.
	For optimizing B, swap any mention of A and B, and pass the data in column-sparse format */
void pgd_iteration(double *A, double *B, double *Xr, size_t *Xr_indptr, size_t *Xr_indices, size_t dimA, size_t k, size_t nnz,
	double cnst_div, double *cnst_sum, double step_size, size_t npass, int ncores)
{
	int k_int = (int) k;
	size_t nnz_this;

	#ifdef _OPENMP
		#if _OPENMP < 20080101 /* OpenMP < 3.0 */
			size_t ia;
		#endif
	#endif

	#pragma omp parallel for schedule(dynamic) num_threads(ncores) shared(A) private(nnz_this) firstprivate(B, k, k_int, nnz, cnst_sum, cnst_div, npass, Xr, Xr_indptr, Xr_indices)
	for (size_t_for ia = 0; ia < dimA; ia++){

		nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];
		for (size_t p = 0; p < npass; p++){
			calc_grad(buffer_arr, A + ia*k, B, Xr + Xr_indptr[ia], Xr_indices + Xr_indptr[ia], nnz_this, k_int);
			cblas_daxpy(k_int, step_size, buffer_arr, 1, A + ia*k, 1);

			cblas_daxpy(k_int, 1, cnst_sum, 1, A + ia*k, 1);
			cblas_dscal(k_int, cnst_div, A + ia*k, 1);
			for (size_t i = 0; i < k; i++) {A[ia*k + i] = nonneg(A[ia*k + i]);}
		}

	}
}

/* Main function
	A                           : Pointer to the already-initialized A matrix (user-factor)
	Xr, Xr_indptr, Xr_indices   : Pointers to the X matrix in row-sparse format
	B                           : Pointer to the already-initialized B matrix (item-factor)
	Xc, Xc_indptr, Xc_indices   : Pointers to the X matrix in column-sparse format
	dimA                        : Number of rows in the A matrix
	dimB                        : Number of rows in the B matrix
	nnz                         : Number of non-zero elements in the X matrix
	k                           : Dimensionality for the factorizing matrices (number of columns of A and B matrices)
	reg_param                   : Regularization pameter for the L2 norm of the A and B matrices
	step_size                   : Initial step size for PGD updates (will be decreased by 1/2 every iteration)
	numiter                     : Number of iterations for which to run the procedure
	npass                       : Number of updates to the same matrix per iteration
	ncores                      : Number of threads to use
Matrices A and B are optimized in-place.
Function does not have a return value.
*/
void run_poismf(
	double *restrict A, double *restrict Xr, size_t *restrict Xr_indptr, size_t *restrict Xr_indices,
	double *restrict B, double *restrict Xc, size_t *restrict Xc_indptr, size_t *restrict Xc_indices,
	const size_t dimA, const size_t dimB, const size_t nnz, const size_t k,
	const double reg_param, double step_size,
	const size_t numiter, const size_t npass, const int ncores)
{

	double *cnst_sum = (double*) malloc(sizeof(double) * k);
	#pragma omp parallel
	{
		buffer_arr = (double*) malloc(sizeof(double) * k);
	}
	double cnst_div;
	int k_int = (int) k;
	double neg_step_sz = -step_size;

	for (size_t fulliter = 0; fulliter < numiter; fulliter++){

		/* Constants to use later */
		cnst_div = 1 / (1 + 2 * reg_param * step_size);
		sum_by_cols(cnst_sum, B, dimB, k, ncores);
		cblas_dscal(k_int, neg_step_sz, cnst_sum, 1);

		pgd_iteration(A, B, Xr, Xr_indptr, Xr_indices, dimA, k, nnz, cnst_div, cnst_sum, step_size, npass, ncores);


		/* Same procedure repeated for the B matrix */
		sum_by_cols(cnst_sum, A, dimA, k, ncores);
		cblas_dscal(k_int, neg_step_sz, cnst_sum, 1);

		pgd_iteration(B, A, Xc, Xc_indptr, Xc_indices, dimB, k, nnz, cnst_div, cnst_sum, step_size, npass, ncores);

		/* Decrease step size after taking PGD steps in both matrices */
		step_size *= 0.5;
		neg_step_sz = -step_size;

	}

	free(cnst_sum);
	#pragma omp parallel
	{
		free(buffer_arr);
	}
}

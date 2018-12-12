 /* Poisson Factorization for sparse matrices

 Based on alternating proximal gradient iteration.
 Variables must be initialized from outside the main function provided here.
 Writen for C99 standard.

 Copyright David Cortes 2018 */

#include <string.h>

/* BLAS functions
https://stackoverflow.com/questions/52905458/link-cython-wrapped-c-functions-against-blas-from-numpy/52913120#52913120
 */
double ddot(int *N, double *DX, int *INCX, double *DY, int *INCY);
void daxpy(int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY);
void dscal(int *N, double *DA, double *DX, int *INCX);


#ifndef ddot
double ddot_(int *N, double *DX, int *INCX, double *DY, int *INCY);
#define ddot(N, DX, INCX, DY, INCY) ddot_(N, DX, INCX, DY, INCY)
#endif

#ifndef daxpy
void daxpy_(int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY);
#define daxpy(N, DA, DX, INCX, DY, INCY) daxpy_(N, DA, DX, INCX, DY, INCY)
#endif

#ifndef dscal
void dscal_(int *N, double *DA, double *DX, int *INCX);
#define dscal(N, DA, DX, INCX) dscal_(N, DA, DX, INCX)
#endif


/* Aliasing for compiler optimizations */
#ifndef restrict
#ifdef __restrict
#define restrict __restrict
#else
#define restrict
#endif
#endif

/* Visual Studio as of 2018 is stuck with OpenMP 2.0 (released 2002),
   which doesn't support parallel loops with unsigned iterators.
   As the code is wrapped in Cython and Cython does not support typdefs conditional on compiler,
   this will map size_t to long on Windows regardless of compiler.
   Can be safely removed if not compiling with MSVC. */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#define size_t long
#else
#include <stddef.h>
#endif

/* Helper functions */
#define nonneg(x) (x >= 0)? x : 0

void sum_by_cols(double *restrict out, double *restrict M, size_t nrow, size_t ncol, int ncores)
{
	memset(out, 0, sizeof(double) * ncol);

	#pragma omp parallel for schedule(static) num_threads(ncores) firstprivate(nrow, ncol, M) reduction(+:out[:ncol])
	for (size_t row = 0; row < nrow; row++){
		for (size_t col = 0; col < ncol; col++){
			out[col] += M[row*ncol + col];
		}
	}
}

void calc_grad(double *out, double *curr, double *F, double *X, size_t *Xind, size_t nnz_this, int k)
{
	double cnst;
	int one = 1;
	memset(out, 0, sizeof(double) * k);

	for (size_t i = 0; i < nnz_this; i++){
		cnst = X[i] / ddot(&k, F + Xind[i]*k, &one, curr, &one);
		daxpy(&k, &cnst, F + Xind[i]*k, &one, out, &one);
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

	/* DAMN YOU MS: Visual Studio as of 2018 still does not support variable length arrays as set in C standard 20 years ago*/
	#ifndef _MSC_VER
	double buffer1[k];
	double cnst_sum[k];
	#else
	double *buffer1 = (double*) malloc(sizeof(double) * k);
	double *cnst_sum = (double*) malloc(sizeof(double) * k);
	#endif
	double cnst_div;

	int k_int = (int) k;
	size_t nnz_this;

	int one = 1;
	double one_dbl = 1;
	double neg_step_sz = -step_size;

	for (size_t fulliter = 0; fulliter < numiter; fulliter++){

		/* Constants to use later */
		memset(cnst_sum, 0, sizeof(double) * k);
		sum_by_cols(cnst_sum, B, dimB, k, ncores);
		dscal(&k_int, &neg_step_sz, cnst_sum, &one);
		cnst_div = 1 / (1 + 2 * reg_param * step_size);

		#pragma omp parallel for schedule(dynamic) num_threads(ncores) shared(A) private(buffer1, nnz_this) firstprivate(B, k, k_int, nnz, cnst_sum, cnst_div, one, one_dbl, npass, Xr, Xr_indptr, Xr_indices)
		for (size_t ia = 0; ia < dimA; ia++){

			nnz_this = Xr_indptr[ia + 1] - Xr_indptr[ia];
			for (size_t p = 0; p < npass; p++){
				calc_grad(buffer1, A + ia*k, B, Xr + Xr_indptr[ia], Xr_indices + Xr_indptr[ia], nnz_this, k_int);
				daxpy(&k_int, &step_size, buffer1, &one, A + ia*k, &one);

				daxpy(&k_int, &one_dbl, cnst_sum, &one, A + ia*k, &one);
				dscal(&k_int, &cnst_div, A + ia*k, &one);
				for (size_t i = 0; i < k; i++) {A[ia*k + i] = nonneg(A[ia*k + i]);}
			}

		}

		/* Same procedure repeated for the B matrix */
		/* Constants to use later */
		memset(cnst_sum, 0, sizeof(double) * k);
		sum_by_cols(cnst_sum, A, dimA, k, ncores);
		dscal(&k_int, &neg_step_sz, cnst_sum, &one);

		#pragma omp parallel for schedule(dynamic) num_threads(ncores) shared(B) private(buffer1, nnz_this) firstprivate(A, k, k_int, nnz, cnst_sum, cnst_div, one, one_dbl, npass, Xc, Xc_indptr, Xc_indices)
		for (size_t ib = 0; ib < dimB; ib++){

			nnz_this = Xc_indptr[ib + 1] - Xc_indptr[ib];
			for (size_t p = 0; p < npass; p++){
				calc_grad(buffer1, B + ib*k, A, Xc + Xc_indptr[ib], Xc_indices + Xc_indptr[ib], nnz_this, k_int);
				daxpy(&k_int, &step_size, buffer1, &one, B + ib*k, &one);

				daxpy(&k_int, &one_dbl, cnst_sum, &one, B + ib*k, &one);
				dscal(&k_int, &cnst_div, B + ib*k, &one);
				for (size_t i = 0; i < k; i++) {B[ib*k + i] = nonneg(B[ib*k + i]);}
			}
		}

		step_size *= 0.5;
		neg_step_sz = -step_size;

	}

	#ifdef _MSC_VER
	free(buffer1);
	free(cnst_sum);
	#endif
}

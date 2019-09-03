/*	Non-negative conjugate gradient optimizer
	
	Minimizes a function subject to non-negativity constraints on all the variables,
	using a modified Polak-Ribiere-Polyak conjugate gradient method. Implementation
	is based on the paper:

	Li, C. (2013). A conjugate gradient type method for the nonnegative constraints optimization problems. Journal of Applied Mathematics, 2013.

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


/*	Note: this file comes from package 'nonneg_cg'
	https://github.com/david-cortes/nonneg_cg
	Visit the link above for the latest version.
	The file was copied here because, in the Python version,
	a) It's too complicated to use functions from another package
	   directly in C in Windows OS (in Linux and Mac this is easy
	   though, and in R it's possible with some constraints).
	b) Some versions of NumPy will throw problems with writing
	   to some non-existent standard output stream when one
	   tries to install a dependency package from 'pip' which
	   needs compilation. This only happens in Windows OS (again)
	   and with some NumPy versions. It's a recurring bug that
	   has been fixed and reintroduced more than once.
	This file has undergone some modifications from the original
	as its use in here is more limited.
*/


#ifdef __cplusplus
extern "C" {
#endif

#ifndef _FOR_R /* not used in the R version */

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#ifndef _FOR_R
	#include <stdio.h> 
#else
	#include <R_ext/Print.h>
	#define printf Rprintf
	#define fprintf(f, message) REprintf(message)
#endif
#ifdef _OPENMP
	#include <omp.h>
#endif

#ifdef _FOR_PYTHON
	// #include "findblas.h"
	double cblas_ddot(int n, double *x, int incx, double *y, int incy);
	void cblas_daxpy(int n, double a, double *x, int incx, double *y, int incy);
	void cblas_dscal(int n, double alpha, double *x, int incx);
#elif defined(_FOR_R)
	#include <R_ext/BLAS.h>
	double cblas_ddot(int n, double *x, int incx, double *y, int incy) { return ddot_(&n, x, &incx, y, &incy); }
	void cblas_daxpy(int n, double a, double *x, int incx, double *y, int incy) { daxpy_(&n, &a, x, &incx, y, &incy); }
	void cblas_dscal(int n, double alpha, double *x, int incx) { dscal_(&n, &alpha, x, &incx); }
#else
	#include "blasfuns.h"
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

/*	OpenMP < 3.0 (e.g. MSVC as of 2019) does not support parallel for's with unsigned iterators,
	and does not support declaring the iterator type in the loop itself */
#ifdef _OPENMP
	#if _OPENMP > 200801 /* OpenMP < 3.0 */
		#define size_t_for size_t
	#else
		#define size_t_for size_t
	#endif
#else
	#define size_t_for size_t
#endif

#ifndef isnan
	#ifdef _isnan
		#define isnan _isnan
	#else
		#define isnan(x) ( (x) != (x) )
	#endif
#endif
#ifndef isinf
	#ifdef _finite
		#define isinf(x) (!_finite(x))
	#else
		#define isinf(x) ( (x) >= HUGE_VAL || (x) <= -HUGE_VAL )
	#endif
#endif

#define get_curr_ix_rotation(ix, n) (  ((ix) == 0) ? 0 : (n)  )
#define incr_ix_rotation(ix) (  ((ix) == 0)? 1 : 0  )
#define square(x) ( (x) * (x) )
#define nonneg(x) ((x) > 0)? (x) : 0

typedef void fun_eval(double x[], int n, double *f, void *data);
typedef void grad_eval(double x[], int n, double grad[], void *data);
typedef void callback(double x[], int n, double f, size_t iter, void *data);

typedef enum cg_result {tol_achieved = 0, stop_maxnfeval = 1, stop_maxiter = 2, out_of_mem = 3} cg_result;

/*	Non-negative conjugate gradient optimizer
	
	Minimizes a function subject to non-negativity constraints on all the variables,
	using a modified Polak-Rubiere-Polyak conjugate gradient method. Implementation
	is based on the paper:

	Li, C. (2013). A conjugate gradient type method for the nonnegative constraints optimization problems. Journal of Applied Mathematics, 2013.
	
	x (in, out)		: At input, starting point (must be a feasible point). At output, optimal values calculated by the optimizer.
	n 				: Number of variables in the optimization problem
	fun_val (out)	: Value of the function achieved at the end of the procedure
	obj_fun			: function that calculates the objective value (must be written into the *f pointer passed to it)
	grad_fun		: function that calculates the gradient (must be written into the grad[] array passed to it)
	cb				: callback function to execute at the end of each iteration
	data			: Extra data to pass to the functions that evaluate objective, gradient, and callback (must be cast to void pointer)
	tol				: Tolerance for <gradient, direction>
					  (Recommended: <1e-3)
	maxnfeval		: Maximum number of function evaluations
					  (Recommended: >1000)
	maxiter			: Maximum number of CG iterations to run
					  (Recommended: >100, but note that steps are always feasible descent directions)
	niter (out)		: Number of CG iterations performed
	nfeval (out)	: Number of function evaluations performed
	decr_lnsrch		: Number by which to decrease the step size after each unsuccessful line search
					  (Recommended: 0.5)
	lnsrch_const	: Acceptance parameter for the line search procedure
					  (Recommended: 0.01)
	max_ls			: Maximum number of line search trials per iteration
					  (Recommended: 20)
	extra_nonneg_tol: Ensure extra non-negative tolerance by explicitly setting elements that are <=0 to zero at each iteration
					  (Recommended: 0)
	buffer_arr		: Array of dimensions (4*n). Will allocate it and then free it if passing NULL.
	nthreads		: Number of parallel threads to use
	verbose			: Whether to print convergence messages
*/
int minimize_nonneg_cg(double x[restrict], int n, double *fun_val,
	fun_eval *obj_fun, grad_eval *grad_fun, callback *cb, void *data,
	double tol, size_t maxnfeval, size_t maxiter, size_t *niter, size_t *nfeval,
	double decr_lnsrch, double lnsrch_const, size_t max_ls,
	int extra_nonneg_tol, double *buffer_arr, int nthreads, int verbose)
{
	double max_step;
	double direction_norm_sq;
	double grad_prev_norm_sq;
	double prod_grad_dir;
	double theta;
	double beta;
	double curr_fun_val;
	double new_fun_val;
	obj_fun(x, n, &curr_fun_val, data);
	*nfeval = 1;
	int dealloc_buffer = 0;
	int revert_x = 0;
	size_t ls;
	cg_result return_value = stop_maxiter;
	if ( maxiter <= 0 ) { maxiter = INT_MAX;}
	if ( maxnfeval <= 0 ) { maxnfeval = INT_MAX;}

	// #if defined(_OPENMP) && (_OPENMP < 200801)  OpenMP < 3.0 
	// long i;
	// long n_szt = n;
	// #else
	size_t n_szt = (size_t) n;
	// #endif

	/*	algorithm requires current and previous gradient and search direction, so the index
		at which they are written in the array is rotated each iteration to avoid unnecessary copies */
	int ix_rotation = 0;
	if (buffer_arr == NULL)
	{
		buffer_arr = (double*) malloc(sizeof(double) * n * 4);
		dealloc_buffer = 1;
		if (buffer_arr == NULL)
		{
			fprintf(stderr, "Could not allocate memory for optimization procedure\n");
			return out_of_mem;
		}
	}
	double *grad_curr_n_prev = buffer_arr;
	double *direction_curr_n_prev = buffer_arr + 2 * n;
	double *restrict direction_curr = direction_curr_n_prev;
	double *restrict grad_curr = grad_curr_n_prev;
	double *restrict direction_prev;
	double *restrict grad_prev;

	// /* set number of BLAS threads */
	#if defined(mkl_set_num_threads_local)
		int ignore = mkl_set_num_threads_local(nthreads);
	#elif defined(openblas_set_num_threads)
		openblas_set_num_threads(nthreads);
	#elif defined(_OPENMP)
		omp_set_num_threads(nthreads);
	#endif


	// if (verbose)
	// {
	// 	printf("********************************************\n");
	// 	printf("Non-negative Conjugate Gradient Optimization\n\n");
	// 	printf("Number of variables to optimize: %d\n", n);
	// 	if (maxiter == INT_MAX && maxnfeval == INT_MAX) {printf("[Warning: no limit on iterations and function evaluations passed]");}
	// 	printf("Initial function value: %10.4f\n\n", curr_fun_val);
	// }

	for (*niter = 0; *niter < maxiter; (*niter)++)
	{
		/* get gradient */
		grad_fun(x, n, grad_curr, data);

		/* determine search direction - this requires 3 passess over 'x' */

		/* first pass: get a capped gradient */
		// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, direction_curr, grad_curr) num_threads(nthreads)
		for (size_t_for i = 0; i < n_szt; i++)
		{
			direction_curr[i] = (x[i] <= 0 && grad_curr[i] >= 0)? 0 : -grad_curr[i];
		}

		/* at first iteration, stop with that */
		if (*niter > 0)
		{
			/* second pass: calculate beta and theta constants */
			theta = 0;
			beta = 0;
			// #if !defined(_WIN32) && !defined(_WIN64)
			// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, direction_prev, grad_curr, grad_prev, n_szt) reduction(+:theta, beta) num_threads(nthreads)
			// #endif
			for (size_t_for i = 0; i < n_szt; i++)
			{
				theta += ( x[i] <= 0 )? 0 : grad_curr[i] * direction_prev[i];
				beta += ( x[i] <= 0 )? 0 : grad_curr[i] * (grad_curr[i] - grad_prev[i]);
			}
			theta /= grad_prev_norm_sq;
			beta /= grad_prev_norm_sq;

			/* third pass: add to direction info on previous direction and gradient differences */
			// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, direction_curr, direction_prev, grad_curr, grad_prev, n_szt, theta, beta) num_threads(nthreads)
			for (size_t_for i = 0; i < n_szt; i++)
			{
				direction_curr[i] += ( x[i] <= 0 )? 0 : beta * direction_prev[i] - theta * (grad_curr[i] - grad_prev[i]);
			}

		}

		/* check if stop criterion is satisfied */
		prod_grad_dir = cblas_ddot(n, grad_curr, 1, direction_curr, 1);
		if ( fabs(prod_grad_dir) <= tol )
		{
			return_value = tol_achieved;
			goto terminate_procedure;
		}

		/* determine maximum step size */
		max_step = 1.0;
		// #if defined(_OPENMP)
		// #if !defined(_WIN32) && !defined(_WIN64)
		// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, direction_curr, n_szt) reduction(min: max_step) num_threads(nthreads)
		// #endif
		// for (size_t_for i = 0; i < n_szt; i++)
		// {
		// 	max_step = (direction_curr[i] < 0)? -x[i] / direction_curr[i] : 1.0;
		// }
		// max_step = fmin(max_step, 1.0);

		// #else
		for (size_t i = 0; i < n_szt; i++)
		{
			if (direction_curr[i] < 0) { max_step = fmin(max_step, -x[i] / direction_curr[i]); }
		}
		// #endif

		/* perform line search */
		cblas_daxpy(n, max_step, direction_curr, 1, x, 1);
		direction_norm_sq = cblas_ddot(n, direction_curr, 1, direction_curr, 1);
		for (ls = 0; ls < max_ls; ls++)
		{
			if (extra_nonneg_tol)
			{
				// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, n_szt) num_threads(nthreads)
				for (size_t_for i = 0; i < n_szt; i++){x[i] = nonneg(x[i]);}
			}
			obj_fun(x, n, &new_fun_val, data);
			if ( !isinf(new_fun_val) && !isnan(new_fun_val) )
			{
				if (new_fun_val <=  curr_fun_val - lnsrch_const * square(max_step * pow(decr_lnsrch, ls)) * direction_norm_sq)
					{ break; }
			}
			(*nfeval)++; if (*nfeval >= maxnfeval) { revert_x = 1; return_value = stop_maxnfeval; goto terminate_procedure; }
			/* go to new step size by modifying x in-place */
			cblas_daxpy(n, max_step * ( pow(decr_lnsrch, ls + 1) - pow(decr_lnsrch, ls) ), direction_curr, 1, x, 1);
		}
		curr_fun_val = new_fun_val;
		if ( cb != NULL) { cb(x, n, curr_fun_val, *niter, data); }

		/* update norm of gradient */
		grad_prev_norm_sq = cblas_ddot(n, grad_curr, 1, grad_curr, 1);

		/* next time, write to the other side of grad and dir arrays */
		direction_prev = direction_curr;
		grad_prev = grad_curr;
		ix_rotation = incr_ix_rotation(ix_rotation);
		direction_curr = direction_curr_n_prev + get_curr_ix_rotation(ix_rotation, n);
		grad_curr = grad_curr_n_prev + get_curr_ix_rotation(ix_rotation, n);
		// if (verbose)
		// {
		// 	printf("Iteration %3d : f(x) = %10.4f, |<g(x), d(x)>| = %12.4f, nfev = %3d, ls = %2d\n",
		// 			(int) *niter + 1, curr_fun_val, fabs(prod_grad_dir), (int) *nfeval, (int) ls +1 );
		// }
	}

	terminate_procedure:
		if (dealloc_buffer) { free(buffer_arr); }
		if (revert_x)
		{
			cblas_daxpy(n, -max_step * pow(decr_lnsrch, ls), direction_curr, 1, x, 1);
			if (extra_nonneg_tol)
			{
				// #pragma omp parallel for schedule(static, n/nthreads) firstprivate(x, n_szt) num_threads(nthreads)
				for (size_t_for i = 0; i < n_szt; i++){x[i] = nonneg(x[i]);}
			}
		}
		// if (verbose)
		// {
		// 	if (return_value == tol_achieved) 	{ printf("\nTerminated: |<g(x), d(x)>| driven below tol.\n"); }
		// 	if (return_value == stop_maxnfeval) { printf("\nTerminated: reached maximum number of function evaluations\n"); }
		// 	if (return_value == stop_maxiter) 	{ printf("\nTerminated: reached maximum number of iterations\n"); }
		// 	printf("Last f(x) = %10.4f\n\n", curr_fun_val);
		// }
		*fun_val = curr_fun_val;
	return (int) return_value;
}


#endif /* _FOR_R */

#ifdef __cplusplus
}
#endif

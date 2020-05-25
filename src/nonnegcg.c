/*  
    Non-negative conjugate gradient optimizer
    
    Minimizes a function subject to non-negativity constraints on all the
    variables, using a modified Polak-Ribiere-Polyak conjugate gradient method.

    Implementation is based on the paper:
        Li, C. (2013).
        "A conjugate gradient type method for the nonnegative constraints optimization problems."
        Journal of Applied Mathematics, 2013.

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


/*  Note: this file comes from package 'nonneg_cg'
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

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <stdbool.h>
    #include <string.h>
#ifndef _FOR_R
    #include <stdio.h> 
#else
    #include <R_ext/Print.h>
    #define printf Rprintf
    #define fprintf(f, message) REprintf(message)
#endif


double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
void cblas_dscal(const int N, const double alpha, double *X, const int incX);

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

typedef void fun_eval(double x[], int n, double *f, void *data);
typedef void grad_eval(double x[], int n, double grad[], void *data);
typedef void callback(double x[], int n, double f, size_t iter, void *data);

typedef enum cg_result {tol_achieved = 0, stop_maxnfeval  = 1,
                        stop_maxiter = 2, unsuccessful_ls = 3,
                        out_of_mem   = 4} cg_result;

/*  Non-negative conjugate gradient optimizer
    
    Minimizes a function subject to non-negativity constraints on all the variables,
    using a modified Polak-Ribiere-Polyak conjugate gradient method. Implementation
    is based on the paper:

    Li, C. (2013). A conjugate gradient type method for the nonnegative constraints optimization problems. Journal of Applied Mathematics, 2013.
    
    x (in, out)     : At input, starting point (must be a feasible point). At output, optimal values calculated by the optimizer.
    n               : Number of variables in the optimization problem
    fun_val (out)   : Value of the function achieved at the end of the procedure
    obj_fun         : function that calculates the objective value (must be written into the *f pointer passed to it)
    grad_fun        : function that calculates the gradient (must be written into the grad[] array passed to it)
    cb              : callback function to execute at the end of each iteration
    data            : Extra data to pass to the functions that evaluate objective, gradient, and callback (must be cast to void pointer)
    tol             : Tolerance for <gradient, direction>
                      (Recommended: <1e-3)
    maxnfeval       : Maximum number of function evaluations
                      (Recommended: >1000)
    maxiter         : Maximum number of CG iterations to run
                      (Recommended: >100, but note that steps are always feasible descent directions)
    niter (out)     : Number of CG iterations performed
    nfeval (out)    : Number of function evaluations performed
    decr_lnsrch     : Number by which to decrease the step size after each unsuccessful line search
                      (Recommended: 0.5)
    lnsrch_const    : Acceptance parameter for the line search procedure
                      (Recommended: 0.01)
    max_ls          : Maximum number of line search trials per iteration
                      (Recommended: 20)
    limit_step      : Whether to limit the step sizes so as to make at most 1
                      variable become zero after each update - this is the strategy
                      prescribed in the reference paper
                      (Recommended: true)
    buffer_arr      : Array of dimensions (5*n). Will allocate it and then free it if passing NULL.
    nthreads        : Number of parallel threads to use
    verbose         : Whether to print convergence messages
*/
int minimize_nonneg_cg(double *restrict x, int n, double *fun_val,
    fun_eval *obj_fun, grad_eval *grad_fun, callback *cb, void *data,
    double tol, size_t maxnfeval, size_t maxiter, size_t *niter, size_t *nfeval,
    double decr_lnsrch, double lnsrch_const, size_t max_ls,
    bool limit_step, double *buffer_arr, int nthreads, int verbose)
{
    double max_step;
    double direction_norm_sq;
    double grad_prev_norm_sq = 0;
    double prod_grad_dir;
    double theta;
    double beta;
    double curr_fun_val;
    double new_fun_val;
    obj_fun(x, n, &curr_fun_val, data);
    *nfeval = 1;
    bool dealloc_buffer = false;
    double curr_step;
    size_t ls;
    cg_result return_value = stop_maxiter;
    if ( maxiter <= 0 ) { maxiter = INT_MAX;}
    if ( maxnfeval <= 0 ) { maxnfeval = INT_MAX;}

    size_t n_szt = (size_t) n;

    /*  algorithm requires current and previous gradient and search direction, so the index
        at which they are written in the array is rotated each iteration to avoid unnecessary copies */
    int ix_rotation = 0;
    if (buffer_arr == NULL)
    {
        buffer_arr = (double*) malloc(sizeof(double) * n * 5);
        dealloc_buffer = true;
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
    double *restrict direction_prev = NULL;
    double *restrict grad_prev = NULL;
    double *restrict new_x = buffer_arr + 4 * n;

    if (isnan(curr_fun_val) || isinf(curr_fun_val)) {
        return_value = unsuccessful_ls;
        goto terminate_procedure;
    }

    for (*niter = 0; *niter < maxiter; (*niter)++)
    {
        /* get gradient */
        grad_fun(x, n, grad_curr, data);

        /* determine search direction - this requires 3 passess over 'x' */

        /* first pass: get a capped gradient */
        for (size_t i = 0; i < n_szt; i++)
        {
            direction_curr[i] = (x[i] <= 0. && grad_curr[i] >= 0.)? 0. : -grad_curr[i];
        }

        /* at first iteration, stop with that */
        if (*niter > 0)
        {
            /* second pass: calculate beta and theta constants */
            theta = 0;
            beta = 0;
            for (size_t i = 0; i < n_szt; i++)
            {
                theta += ( x[i] <= 0. )? 0. : grad_curr[i] * direction_prev[i];
                beta  += ( x[i] <= 0. )? 0. : grad_curr[i] * (grad_curr[i] - grad_prev[i]);
            }
            theta /= grad_prev_norm_sq;
            beta /= grad_prev_norm_sq;

            /* third pass: add to direction info on previous direction and gradient differences */
            for (size_t i = 0; i < n_szt; i++)
            {
                direction_curr[i] += ( x[i] <= 0. )? 0. : beta * direction_prev[i] - theta * (grad_curr[i] - grad_prev[i]);
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
        if (limit_step)
        {
            max_step = 1.;
            for (size_t i = 0; i < n_szt; i++) {
                if (direction_curr[i] < 0.)
                    max_step = fmin(max_step, -x[i] / direction_curr[i]);
            }
        }

        else {
            max_step = 0.;
            for (size_t i = 0; i < n_szt; i++) {
                if (direction_curr[i] < 0.)
                    max_step = fmax(max_step, -x[i] / direction_curr[i]);
            }
            max_step = fmin(1., 0.99 * max_step);
        }

        /* perform line search */
        direction_norm_sq = cblas_ddot(n, direction_curr, 1, direction_curr, 1);
        curr_step = max_step;
        for (ls = 0; ls < max_ls; ls++)
        {
            memcpy(new_x, x, n*sizeof(double));
            cblas_daxpy(n, curr_step, direction_curr, 1, new_x, 1);
            if (limit_step)
                for (size_t i = 0; i < n_szt; i++)
                    new_x[i] = (new_x[i] >= 1e-15)? new_x[i] : 0.;
            else
                for (size_t i = 0; i < n_szt; i++)
                    new_x[i] = (new_x[i] > 0.)? new_x[i] : 0.;
            obj_fun(new_x, n, &new_fun_val, data);
            if ( !isinf(new_fun_val) && !isnan(new_fun_val) )
            {
                if (
                    new_fun_val <= 
                    curr_fun_val - lnsrch_const * curr_step * direction_norm_sq
                    )
                    { memcpy(x, new_x, n*sizeof(double)); break; }
            }
            (*nfeval)++;
            if (*nfeval >= maxnfeval) {
                return_value = stop_maxnfeval;
                goto terminate_procedure;
            }
            if (ls == max_ls + 1) {
                new_fun_val = curr_fun_val;
                return_value = unsuccessful_ls;
                goto terminate_procedure;
            }
            curr_step *= decr_lnsrch;
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
    }

    terminate_procedure:
        if (dealloc_buffer) { free(buffer_arr); }
        *fun_val = curr_fun_val;
    return (int) return_value;
}


#ifdef __cplusplus
}
#endif

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

bool check_is_sorted(sparse_ix arr[], size_t n)
{
    if (n <= 1) return true;
    for (sparse_ix ix = 0; ix < n-1; ix++)
        if (arr[ix] > arr[ix+1]) return false;
    return true;
}

/* https://www.stat.cmu.edu/~ryantibs/median/quickselect.c */
/* Some sample C code for the quickselect algorithm, 
   taken from Numerical Recipes in C. */
#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void qs_argpartition(sparse_ix arr[], real_t values[], size_t n, size_t k)
{
    sparse_ix i,ir,j,l,mid;
    size_t a,temp;

    l=0;
    ir=n-1;
    for(;;) {
        if (ir <= l+1) { 
            if (ir == l+1 && values[arr[ir]] > values[arr[l]]) {
                SWAP(arr[l],arr[ir]);
            }
            return;
        }
        else {
            mid=(l+ir) >> 1; 
            SWAP(arr[mid],arr[l+1]);
            if (values[arr[l]] < values[arr[ir]]) {
                SWAP(arr[l],arr[ir]);
            }
            if (values[arr[l+1]] < values[arr[ir]]) {
                SWAP(arr[l+1],arr[ir]);
            }
            if (values[arr[l]] < values[arr[l+1]]) {
                SWAP(arr[l],arr[l+1]);
            }
            i=l+1; 
            j=ir;
            a=arr[l+1]; 
            for (;;) { 
                do i++; while (values[arr[i]] > values[a]); 
                do j--; while (values[arr[j]] < values[a]); 
                if (j < i) break; 
                SWAP(arr[i],arr[j]);
            } 
            arr[l+1]=arr[j]; 
            arr[j]=a;
            if (j >= k) ir=j-1; 
            if (j <= k) l=i;
        }
    }
}

int cmp_size_t(const void *a, const void *b)
{
    return *((sparse_ix*)a) - *((sparse_ix*)b);
}

real_t *ptr_real_t_glob = NULL;
#pragma omp threadprivate(ptr_real_t_glob)
int cmp_argsort(const void *a, const void *b)
{
    real_t v1 = ptr_real_t_glob[*((sparse_ix*)a)];
    real_t v2 = ptr_real_t_glob[*((sparse_ix*)b)];
    return (v1 == v2)? 0 : ((v1 < v2)? 1 : -1);
}

int topN
(
    real_t *restrict a_vec, real_t *restrict B, int k,
    sparse_ix *restrict include_ix, size_t n_include,
    sparse_ix *restrict exclude_ix, size_t n_exclude,
    sparse_ix *restrict outp_ix, real_t *restrict outp_score,
    size_t n_top, size_t n, int nthreads
)
{
    if (n_include == 0) include_ix = NULL;
    if (n_exclude == 0) exclude_ix = NULL;

    if (include_ix != NULL && exclude_ix != NULL)
        return 2;
    if (n_top == 0) return 2;
    if (n_exclude > n-n_top) return 2;
    if (n_include > n) return 2;

    #if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
    long long ix = 0;
    #else
    size_t ix = 0;
    #endif

    int retval = 0;
    size_t n_take = (include_ix != NULL)?
                     (size_t)n_include :
                     ((exclude_ix == NULL)? (size_t)n : (size_t)(n-n_exclude) );

    real_t *restrict buffer_scores = NULL;
    sparse_ix *restrict buffer_ix = NULL;
    sparse_ix *restrict buffer_mask = NULL;

    if (include_ix != NULL) {
        buffer_ix = include_ix;
    }

    else {
        buffer_ix = (sparse_ix*)malloc((size_t)n*sizeof(sparse_ix));
        if (buffer_ix == NULL) { retval = 1; goto cleanup; }
        for (sparse_ix ix = 0; ix < (sparse_ix)n; ix++) buffer_ix[ix] = ix;
    }

    if (exclude_ix != NULL)
    {
        sparse_ix move_to = n-1;
        sparse_ix temp;
        if (!check_is_sorted(exclude_ix, n_exclude))
            qsort(exclude_ix, n_exclude, sizeof(sparse_ix), cmp_size_t);

        for (sparse_ix ix = n_exclude-1; ix >= 0; ix--) {
            temp = buffer_ix[move_to];
            buffer_ix[move_to] = exclude_ix[ix];
            buffer_ix[exclude_ix[ix]] = temp;
            move_to--;
            if (ix == 0) break;
        }
    }

    /* Case 1: there is a potentially small number of items to include.
       Here can produce predictons only for those, then make
       an argsort with doubly-masked indices. */
    if (include_ix != NULL)
    {
        buffer_scores = (real_t*)malloc((size_t)n_include*sizeof(real_t));
        buffer_mask = (sparse_ix*)malloc((size_t)n_include*sizeof(sparse_ix));
        if (buffer_scores == NULL || buffer_mask == NULL) {
            retval = 1;
            goto cleanup;
        }
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(a_vec, B, k, n_include, include_ix, buffer_scores)
        for (ix = 0; ix < (sparse_ix)n_include; ix++) {
            buffer_scores[ix] = cblas_tdot(k, a_vec, 1,
                                           B + (size_t)include_ix[ix] * (size_t)k, 1);
        }
        for (sparse_ix ix = 0; ix < (sparse_ix)n_include; ix++)
            buffer_mask[ix] = ix;
    }

    /* Case 2: there is a large number of items to exclude.
       Here can also produce predictions only for the included ones
       and then make a full or partial argsort. */
    else if (exclude_ix != NULL && (real_t)n_exclude > (real_t)n/20.)
    {
        buffer_scores = (real_t*)malloc(n_take*sizeof(real_t));
        buffer_mask = (sparse_ix*)malloc(n_take*sizeof(sparse_ix));
        if (buffer_scores == NULL || buffer_mask == NULL) {
            retval = 1;
            goto cleanup;
        }
        for (sparse_ix ix = 0; ix < (sparse_ix)n_take; ix++)
            buffer_mask[ix] = ix;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(a_vec, B, k, n_take, buffer_ix, buffer_scores)
        for (ix = 0; ix < (sparse_ix)n_take; ix++)
            buffer_scores[ix] = cblas_tdot(k, a_vec, 1,
                                           B + (size_t)buffer_ix[ix] * (size_t)k, 1);
    }

    /* General case: make predictions for all the entries, then
       a partial argsort (this is faster since it makes use of
       optimized BLAS gemv, but it's not memory-efficient) */
    else
    {
        buffer_scores = (real_t*)malloc((size_t)n*sizeof(real_t));
        if (buffer_scores == NULL) { retval = 1; goto cleanup; }
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    n, k,
                    1., B, k,
                    a_vec, 1,
                    0., buffer_scores, 1);
    }

    /* If there is no real_t-mask for indices, do a partial argsort */
    ptr_real_t_glob = buffer_scores;
    if (buffer_mask == NULL)
    {
        /* If the number of elements is very small, it's faster to
           make a full argsort, taking advantage of qsort's optimizations */
        if (n_take <= 50 || n_take >= (real_t)n*0.75)
        {
            qsort(buffer_ix, n_take, sizeof(sparse_ix), cmp_argsort);
        }

        /* Otherwise, do a proper partial sort */
        else
        {
            qs_argpartition(buffer_ix, buffer_scores, n_take, n_top);
            qsort(buffer_ix, n_top, sizeof(sparse_ix), cmp_argsort);
        }

        memcpy(outp_ix, buffer_ix, (size_t)n_top*sizeof(sparse_ix));
    }

    /* Otherwise, do a partial argsort with doubly-indexed arrays */
    else
    {
        if (n_take <= 50 || n_take >= (real_t)n*0.75)
        {
            qsort(buffer_mask, n_take, sizeof(sparse_ix), cmp_argsort);
        }

        else
        {
            qs_argpartition(buffer_mask, buffer_scores, n_take, n_top);
            qsort(buffer_mask, n_top, sizeof(sparse_ix), cmp_argsort);
        }

        for (sparse_ix ix = 0; ix < (sparse_ix)n_top; ix++)
            outp_ix[ix] = buffer_ix[buffer_mask[ix]];
    }
    ptr_real_t_glob = NULL;

    /* If scores were requested, need to also output those */
    if (outp_score != NULL)
    {
        if (buffer_mask == NULL)
            for (sparse_ix ix = 0; ix < (sparse_ix)n_top; ix++)
                outp_score[ix] = buffer_scores[outp_ix[ix]];
        else
            for (sparse_ix ix = 0; ix < (sparse_ix)n_top; ix++)
                outp_score[ix] = buffer_scores[buffer_mask[ix]];
    }

    cleanup:
        free(buffer_scores);
        if (include_ix == NULL)
            free(buffer_ix);
        free(buffer_mask);
    if (retval == 1) return retval;
    return 0;
} 

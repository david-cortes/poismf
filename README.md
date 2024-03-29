# Poisson Factorization

Fast and memory-efficient non-negative matrix factorization for sparse counts data which produces sparse factor matrices, based on Poisson likelihood with regularization. The method is described in ["Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations"](http://arxiv.org/abs/1811.01908).

The model is similar to [Hierarchical Poisson Factorization](https://arxiv.org/abs/1311.1704), but uses regularization instead of a bayesian hierarchical structure, and is fit through gradient-based methods instead of coordinate ascent. It tries to approximate a sparse matrix of counts as a product of two lower-dimensional matrices in a way that maximizes Poisson likelihood - i.e.:
```
X ~ Poisson(A * t(B))
```

The implementation is in C with interfaces for Python and R.

****************************
[(Example Python notebook here)](http://nbviewer.jupyter.org/github/david-cortes/poismf/blob/master/example/example_poismf_lastfm.ipynb)

# Model description

This library tries to fit a low-rank approximate factorization model in which some sparse `X` matrix of counts data is assumed to be Poisson-distributed, with parameters given by the product of two non-negative and lower-dimensional matrices - that is:
```
X ~ Poisson(A * t(B))
```

The model is fit through maximum likelihood estimation (adding a regularization term on the factor matrices) by alternating between updates to the `A` and `B` matrices, exploiting a shortcut for fast evaluation and optimization of Poisson likelihood when the `A` and `B` matrices are constrained to be non-negative and with no link function.

The intended primary use is for recommender systems, in which users are the rows of the `X` matrices, items the columns, and the non-zero values indicate interactions (e.g. clicks, views, plays, etc.) - the idea being that the items with the highest-predicted value for a given user are the best candidates to recommend.

Nevertheless, can also be used for other domains such as topic modeling or as a general dimensionality reduction model - just take any mention of users as rows or documents and any mention of items as columns or words.

Compared to other models, and depending on the optimization method used, this model has the advantage of producing very sparse user and item factor matrices (e.g. over 90% of the entries being exactly zero), which can be desirable in some situations.

# Installation

* Python

**Note:** requires a C compiler configured for Python. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions.

```
pip install poismf
```
or if that fails:
```
pip install --no-use-pep517 poismf
```

** *
**Note for macOS users:** on macOS, the Python version of this package might compile **without** multi-threading capabilities. In order to enable multi-threading support, first install OpenMP:
```
brew install libomp
```
And then reinstall this package: `pip install --upgrade --no-deps --force-reinstall poismf`.
** *
**IMPORTANT:** the setup script will try to add compilation flag `-march=native`. This instructs the compiler to tune the package for the CPU in which it is being installed (by e.g. using AVX instructions if available), but the result might not be usable in other computers. If building a binary wheel of this package or putting it into a docker image which will be used in different machines, this can be overriden either by (a) defining an environment variable `DONT_SET_MARCH=1`, or by (b) manually supplying compilation `CFLAGS` as an environment variable with something related to architecture. For maximum compatibility (but slowest speed), it's possible to do something like this:

```
export DONT_SET_MARCH=1
pip install poismf
```

or, by specifying some compilation flag for architecture:
```
export CFLAGS="-march=x86-64"
pip install poismf
```
** *

* R

**Note:** This package benefits from extra optimizations that aren't enabled by default for R packages. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions.

```r
install.packages("poismf")
```

It is recommended to use an optimized library for BLAS operations such as MKL or OpenBLAS - see [this post](https://github.com/david-cortes/R-openblas-in-windows) for instructions on getting OpenBLAS in R for Windows. Alternatively, Microsoft's MRAN distribution comes MKL preinstalled.

# Getting started

Example IPython notebook using the package with the Last.FM dataset:

[Notebook](http://nbviewer.jupyter.org/github/david-cortes/poismf/blob/master/example/example_poismf_lastfm.ipynb)


# Sample usage

* Python

(API is very similar to `cmfrec` library)
```python
import numpy as np, pandas as pd

## Generating random sparse data
nusers = 10 ** 2
nitems = 10 ** 3
nnz    = 10 ** 4

np.random.seed(1)
df = pd.DataFrame({
    'UserId' : np.random.randint(nusers, size = nnz),
    'ItemId' : np.random.randint(nitems, size = nnz),
    'Count'  : 1 + np.random.gamma(1, 1, size = nnz).astype(int)
})
### (can also pass a sparse COO matrix instead)

## Fitting the model
from poismf import PoisMF

### good speed, not-so-good quality
model = PoisMF(k=5, method="pg")
### good quality, but slow
model = PoisMF(k=5, method="tncg")
### balance between speed-quality
model = PoisMF(k=5, method="cg")
model.fit(df)

### Predict functionality (chosen entries in X)
model.predict(1, 10) ## entry [1, 10]
model.predict(np.array([1,1,1]), np.array([4,5,6])) ## entries [1,4], [1,5], [1,6]

### Ranking functionality (for recommender systems)
model.topN(user=2, n=5, exclude=df.ItemId.loc[df.UserId==2])
model.topN_new(X=df[["ItemId","Count"]].loc[df.UserId==2],
               n=5, exclude=df.ItemId.loc[df.UserId==2])
## For faster fitting without any checks and castings, can use 'fit_unsafe' too
```

(For a longer example see the [IPython notebook](http://nbviewer.jupyter.org/github/david-cortes/poismf/blob/master/example/example_poismf_lastfm.ipynb) in the section above)

* R

(See example in the documentation for a slightly longer version)

```r
library(poismf)

### Create a random sparse data frame in COO format
nrow <- 10^2
ncol <- 10^3
nnz  <- 10^4
set.seed(1)
X <- data.frame(
    row_ix = sample(nrow, size=nnz, replace=TRUE),
    col_ix = sample(ncol, size=nnz, replace=TRUE),
    count  = rpois(nnz, 1) + 1)
X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
### (can also pass as sparse matrix from Matrix or SparseM)

### Factorize the randomly-generated sparse matrix
### good speed, not-so-good quality
model <- poismf(k=5, X, method="pg")
### good quality, but slow
model <- poismf(k=5, X, method="tncg")
### balance between speed-quality
model <- poismf(k=5, X, method="cg")

### Predict functionality (chosen entries in X)
predict(model, 1, 10) ## entry [1, 10]
predict(model, c(1, 1, 1), c(4, 5, 6)) ## entries [1,4], [1,5], [1,6]

### Ranking functionality (for recommender systems)
topN(model, user=2, n=5, exclude=X$col_ix[X$row_ix==2])
topN.new(model, X=X[X$row_ix==2, c("col_ix","count")],
         n=5, exclude=X$col_ix[X$row_ix==2])
```

* C:

You can also take the C file `poismf/poismf.c` and use it in some language other than Python or R - works with a copy of `X` in row-sparse and another in column-sparse formats.

```c
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
#define sparse_ix size_t
#define real_t double
typedef enum Method {tncg = 1, cg = 2, pg = 3} Method;
int run_poismf(
    real_t *restrict A, real_t *restrict Xr, sparse_ix *restrict Xr_indptr, sparse_ix *restrict Xr_indices,
    real_t *restrict B, real_t *restrict Xc, sparse_ix *restrict Xc_indptr, sparse_ix *restrict Xc_indices,
    const size_t dimA, const size_t dimB, const size_t k,
    const real_t l2_reg, const real_t l1_reg, const real_t w_mult, real_t step_size,
    const Method method, const bool limit_step, const size_t numiter, const size_t maxupd,
    const bool early_stop, const bool reuse_prev,
    const bool handle_interrupt, const int nthreads)
```

# Documentation

* Python: available at [ReadTheDocs](https://poismf.readthedocs.io/en/latest/).

* R: documentation available internally (e.g. `help(poismf::poismf`)). PDF can be download at [CRAN](https://cran.r-project.org/web/packages/poismf/index.html).

* C: documentation available only for the main function, as given in the previous section.

# References

* Cortes, David. "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations." arXiv preprint arXiv:1811.01908 (2018).

* Nash, Stephen G. "Newton-type minimization via the Lanczos method." SIAM Journal on Numerical Analysis 21.4 (1984): 770-788.

* Li, Can. "A conjugate gradient type method for the nonnegative constraints optimization problems." Journal of Applied Mathematics 2013 (2013).

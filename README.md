# Poisson Factorization

Very fast and memory-efficient non-negative matrix factorization for sparse data, based on Poisson likelihood with l2 regularization. The algorithm is described in the paper "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations":
[http://arxiv.org/abs/1811.01908](http://arxiv.org/abs/1811.01908)

The model is similar to [Hierarchical Poisson Factorization](https://arxiv.org/abs/1311.1704), but uses regularization instead of a bayesian hierarchical structure, and is fit through proximal gradient descent instead of variational inference, resulting in a procedure that, for larger datasets, can be more than 400x faster than HPF, and 15x faster than implicit-ALS as implemented in the package [implicit](https://github.com/benfred/implicit).

(Alternatively, can also use L1 regularization or a mixture of L1 and L2, and use a conjugate gradient method instead of proximal gradient)

The implementation is in C with interfaces for Python and R. Parallelization is through OpenMP.

![image](tables/rr_table.png "retailrocket")
![image](tables/ms_table.png "millionsong")

(_Statistics benchmarked on a Skylake server using 16 cores with proximal gradient method_)

# Installation

* Python

Linux, MacOS + GCC, and Windows depending on NumPy:
```pip install poismf```


MacOS without GCC: install OpenMP modules for CLANG (these don't come by default in apple's "special" clang distribution, even though they are part of standard clang), then install with either `pip install poismf` or `python setup.py install`.


Windows with unlucky NumPy version:
Clone or download the repository and then install with `setup.py`, e.g.:

```
git clone https://github.com/david-cortes/poismf.git
cd poismf
python setup.py install
```
(Note that it requires package `findblas`, can usually be installed with `pip install findblas`)

Depending on configuration, in windows you might also try ```python setup.py install --compiler=msvc```.


Requires some BLAS library such as MKL (`pip install mkl-devel`) or OpenBLAS - will attempt to use the same as NumPy is using. Also requires a C compiler such as GCC or Visual Studio (in windows + conda, install Visual Studio Build Tools, and select package MSVC140 in the install options).

For any installation problems, please open an issue in GitHub providing information about your system (OS, BLAS, C compiler) and Python installation.

* R
```r
install.packages("poismf")
```

# Usage

* Python

(API is very similar to `hpfrec`)
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

## Fitting the model -- note that some functions require package 'hpfrec'
## ('pip install hpfrec')
from poismf import PoisMF
model = PoisMF()
model.fit(df)
model.topN(df.UserId.iloc[0], n = 10)
model.predict(df.UserId.iloc[0], df.ItemId.iloc[10])
model.predict(df.UserId.values[np.random.randint(nnz, size = 10)], df.ItemId.values[np.random.randint(nnz, size = 10)])


### Make sure that the parameters (latent factors) didn't end up all-NaN or all-zeros,
### these are in 'model.A' and 'model.B'

## For CG, need to adjust the default parameters, e.g.
## model = PoisMF(use_cg = True, l2_reg = 1e3, niter=5)

## For faster fitting without any checks and castings, can use Cython function directly too
```

* R

```r
library(poismf)

### create a random sparse data frame in COO format
nrow <- 10 ** 2
ncol <- 10 ** 3
nnz  <- 10 ** 4
set.seed(1)
X <- data.frame(
    row_ix = as.integer(runif(nnz, min = 1, max = nrow)),
    col_ix = as.integer(runif(nnz, min = 1, max = ncol)),
    count = rpois(nnz, 1) + 1)
X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]

### factorize the randomly-generated sparse matrix
model <- poismf(X, nthreads = 1)

### predict functionality
predict(model, 1, 10) ## predict entry (1, 10)
predict(model, 1, topN = 10) ## predict top-10 entries "B" for row 1 of "A".
predict(model, c(1, 1, 1), c(4, 5, 6)) ## predict entries [1,4], [1,5], [1,6]
head(predict(model, 1)) ## predict the whole row 1

#all predictions for new row/user/doc
head(predict(model, data.frame(col_ix = c(1,2,3), count = c(4,5,6)) ))
```
(Can also work with sparse matrices instead of data frames)

* C:

You can also take the C file `poismf/pgd.c` and use it in some language other than Python or R - works with a copy of `X` in row-sparse and another in column-sparse formats.

```c
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
	npass                       : Number of updates to the same matrix per iteration (pass >1 for CG)
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
```

# Documentation

* Python: available at [ReadTheDocs](https://poismf.readthedocs.io/en/latest/).

* R: documentation available internally (e.g. `help(poismf::poismf`)). PDF can be download at [CRAN](https://cran.r-project.org/web/packages/poismf/index.html).

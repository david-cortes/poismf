# Poisson Factorization

Very fast and memory-efficient non-negative matrix factorization for sparse data, based on Poisson likelihood with l2 regularization. The algorithm is described in the paper "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations":
[http://arxiv.org/abs/1811.01908](http://arxiv.org/abs/1811.01908)

The model is similar to [Hierarchical Poisson Factorization](https://arxiv.org/abs/1311.1704), but uses regularization instead of a bayesian hierarchical structure, and is fit through proximal gradient descent instead of variational inference, resulting in a procedure that, for larger datasets, can be more than 400x faster than HPF, and 15x faster than implicit-ALS as implemented in the package [implicit](https://github.com/benfred/implicit).

(Alternatively, can also use L1 regularization or a mixture of L1 and L2, and use a conjugate gradient method instead of proximal gradient, which is slower but less likely to fail)

At the moment it does not have a complete API, only a function to optimize the user/item factor matrices in-place - a similar API to [hpfrec](https://www.github.com/david-cortes/hpfrec) will come in the future. The R version has some basic `predict` functionality though.

The implementation is in C with interfaces for Python and R. Parallelization is through OpenMP.

![image](tables/rr_table.png "retailrocket")
![image](tables/ms_table.png "millionsong")

(_Statistics benchmarked on a Skylake server using 16 cores with proximal gradient method_)

# Installation

* Python

Clone or download the repository and then install with `setup.py`, e.g.:

```
git clone https://github.com/david-cortes/poismf.git
cd poismf
python setup.py install
```
(Note that it requires packages `nonnegcg` and  `findblas`, can usually be installed with `pip install nonnegcg findblas`, but sometimes in Windows depending on `numpy` version, `nonnegcg` might have to be installed through `setup.py install` from [here](https://www.github.com/david-cortes/nonneg_cg).)

Requires some BLAS library such as MKL (comes by default in Anaconda) or OpenBLAS - will attempt to use the same as NumPy is using. Also requires a C compiler such as GCC or Visual Studio.

For any installation problems, please open an issue in GitHub providing information about your system (OS, BLAS, C compiler) and Python installation.

* R

Requires package [nonneg.cg](https://www.github.com/david-cortes/nonneg_cg) as dependency, can be installed with:
```r
intall.packages("nonneg.cg")
```

After that, install this package with:
```r
devtools::install_github("david-cortes/poismf")
```

# Usage

* R

See the documentation for usage examples `help(poismf::poismf)`.

* Python

In rough terms, you'll need to first initialize the user-factor and item-factor matrices yourself randomly (e.g. `~ Gamma(1,1)` or `~ Uniform(0,1)`), then run the optimization routine on them, passing the observed data in sparse coordinate format:

```python
import numpy as np
from scipy.sparse import coo_matrix

## Generating random sparse data
nusers = 10**5
nitems = 10**4
nobs = 10**6

## Sparse COO matrix (data, (row, col))
np.random.seed(1)
values = (np.random.gamma(1, 1, size=nobs) + 1).astype('int64') ## this is just to round values, they are casted anyway later
row_id = np.random.randint(nusers, size=nobs)
col_id = np.random.randint(nitems, size=nobs)
X = coo_matrix((values, (row_id, col_id)), shape=(nusers, nitems))

## Initializing paramters
k = 30 ## number of latent factors
np.random.seed(123)
A = np.random.gamma(1, 1, size=(nusers, k)) ## User factors
B = np.random.gamma(1, 1, size=(nitems, k)) ## Item factors

## Fitting the model
from poismf import run_pgd
run_pgd(X, A, B, ncores=1) ## adjust the number of threads/cores accordingly for your computer
## Matrices A and B are optimized in-place

## Full call
run_pgd(X, A, B, use_cg=False, l2_reg=1e9, l1_reg=0, step_size=1e-7, niter=10, npass=1, ncores=1)

## Note: for conjugate gradient, increase 'npass' and decrease 'l2_reg'

## Making predictions
## Predict count of item 10 for user 25
np.dot(A[25], B[10])


### Be sure to check that your A and B matrices don't turn to NaNs or Zeros!!
```

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

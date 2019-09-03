#' @title Factorization of Sparse Counts Matrices through Poisson Likelihood
#' @description Creates a low-rank non-negative factorization of a sparse counts matrix by
#' maximizing Poisson likelihood with L1/L2 regularization, using optimization routines
#' based on proximal gradient iteration.
#' @param X The matrix to factorize. Can be:
#' a) a `data.frame` with 3 columns, containing in this order:
#' row index (starting at 1), column index, count value (the indices can also be character type, in wich
#' case it will enumerate them internally, and will return those same characters from `predict`);
#' b) A sparse matrix in COO format from the `SparseM` package;
#' c) a full matrix (of class `matrix` or `Matrix::dgTMatrix`);
#' d) a sparse matrix from package `Matrix` in triplets format.
#' @param k Dimensionality of the factorization (a.k.a. number of latent factors).
#' @param l1_reg Strength of the l1 regularization
#' @param l2_reg Strength of the l2 regularization.
#' @param niter Number of iterations to run.
#' @param nupd Number of updates per iteration.
#' @param step_size Initial step size to use (proximal gradient only). Will be decreased by 1/2 after each iteration.
#' @param init_type One of "gamma" or "uniform" (How to intialize the factorizing matrices).
#' @param seed Random seed to use for starting the factorizing matrices.
#' @param nthreads Number of parallel threads to use. Passing a negative number will use
#' the maximum available number of threads
#' @references Cortes, David. "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations." arXiv preprint arXiv:1811.01908 (2018).
#' @return An object of class `poismf` with the following fields of interest:
#' @field A : the user/document/row-factor matrix (as a vector, has to be reshaped to (nrows, k)).
#' @field B : the item/word/column-factor matrix (as a vector, has to be reshaped to (ncols, k)).
#' @export
#' @examples 
#' library(poismf)
#' 
#' ### create a random sparse data frame in COO format
#' nrow <- 10 ** 2
#' ncol <- 10 ** 3
#' nnz  <- 10 ** 4
#' set.seed(1)
#' X <- data.frame(
#'     row_ix = as.integer(runif(nnz, min = 1, max = nrow)),
#'     col_ix = as.integer(runif(nnz, min = 1, max = ncol)),
#'     count = rpois(nnz, 1) + 1)
#' X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
#' 
#' ### factorize the randomly-generated sparse matrix
#' model <- poismf(X, nthreads = 1)
#' 
#' ### predict functionality
#' predict(model, 1, 10) ## predict entry (1, 10)
#' predict(model, 1, topN = 10) ## predict top-10 entries "B" for row 1 of "A".
#' predict(model, c(1, 1, 1), c(4, 5, 6)) ## predict entries [1,4], [1,5], [1,6]
#' head(predict(model, 1)) ## predict the whole row 1
#' 
#' #all predictions for new row/user/doc
#' head(predict(model, data.frame(col_ix = c(1,2,3), count = c(4,5,6)) ))
#' @seealso \link{predict.poismf} \link{predict_all}
poismf <- function(X, k = 50, l1_reg = 0, l2_reg = 1e9, niter = 10, nupd = 1, step_size = 1e-7,
				   init_type = "gamma", seed = 1, nthreads = -1) {
	
	### Check input parameters
	if (NROW(niter) > 1 || niter < 1) { stop("'niter' must be a positive integer.") }
	if (NROW(nthreads) > 1 || nthreads < 1) {nthreads <- parallel::detectCores()}
	if (NROW(k) > 1 || k < 1) { stop("'k' must be a positive integer.") }
	if (nupd < 1) {stop("'nupd' must be a positive integer.")}
	if (l1_reg < 0 | l2_reg < 0) {stop("Regularization parameters must be non-negative.")}
	if (init_type != "gamma" & init_type != "uniform") {stop("'init_type' must be one of 'gamma' or 'uniform'.")}
	
	k         <- as.integer(k)
	l1_reg    <- as.numeric(l1_reg)
	l2_reg    <- as.numeric(l2_reg)
	step_size <- as.numeric(step_size)
	niter     <- as.integer(niter)
	nupd      <- as.integer(nupd)
	nthreads  <- as.integer(nthreads)
	
	is_non_int <- FALSE
	
	### Convert X to CSR and CSC
	if ("data.frame" %in% class(X)) {
		
		if (!("integer") %in% class(X[[1]]) & !("numeric" %in% class(X[[1]]))) { is_non_int <- TRUE }
		if (!("integer") %in% class(X[[2]]) & !("numeric" %in% class(X[[2]]))) { is_non_int <- TRUE }
		if (is_non_int) {
			X[[1]]   <- factor(X[[1]])
			X[[2]]   <- factor(X[[2]])
			levels_A <- levels(X[[1]])
			levels_B <- levels(X[[2]])
			X[[1]]   <- as.integer(X[[1]])
			X[[2]]   <- as.integer(X[[2]])
		}
		
		ix_row <- as.integer(X[[1]])
		ix_col <- as.integer(X[[2]])
		xflat  <- as.numeric(X[[3]])
		
		if (any(is.na(ix_row)) || any(is.na(ix_col)) || any(is.na(xflat))) {
			stop("Input contains missing values.")
		}
		if (any(ix_row < 1) | any(ix_col < 1)) {
			stop("First two columns of 'X' must be row/column indices starting at 1.")
		}
		Xcsr <- Matrix::sparseMatrix(i = ix_col, j = ix_row, x = xflat, giveCsparse = TRUE)
		Xcsc <- Matrix::sparseMatrix(i = ix_row, j = ix_col, x = xflat, giveCsparse = TRUE)
	} else if ("dgTMatrix" %in% class(X)) {
		Xcsr <- as(t(X), "sparseMatrix")
		Xcsc <- as(X, "sparseMatrix")
	} else if ("matrix" %in% class(X)) {
		if (any(is.na(X))) { stop("Input contains missing values.") }
		Xcsr <- as(t(X), "sparseMatrix")
		Xcsc <- as(X, "sparseMatrix")
	} else if ("matrix.coo" %in% class(X)) {
		Xcsr <- SparseM::as.matrix.csr(X)
		Xcsc <- SparseM::as.matrix.csc(X)
	} else {
		stop("'X' must be a 'data.frame' with 3 columns, or a matrix (either full or sparse in triplets or compressed).")
	}
	
	### Another check
	if (min(Xcsr) < 0) { stop("'X' contains negative values.") }
	
	### Get dimensions
	dimA <- NCOL(Xcsr)
	dimB <- NROW(Xcsr)
	if ("matrix.csr" %in% class(Xcsr)) {
		nnz <- length(Xcsr@ra)
	} else {
		nnz <- length(Xcsr@x)
	}
	if (nnz < 1) { stop("Input does not contain non-zero values.") }
	
	### Initialize factor matrices
	set.seed(seed)
	if (init_type == "gamma") {
		A <- rgamma(dimA * k, 1)
		B <- rgamma(dimB * k, 1)
	} else {
		A <- runif(dimA * k)
		B <- runif(dimB * k)
	}
	
	### Run optimizer
	if ("matrix.csr" %in% class(Xcsr)) {
		r_wrapper_poismf(A, B, dimA, dimB, k,
						 Xcsr@ra, Xcsr@ja - 1, Xcsr@ia - 1,
						 Xcsc@ra, Xcsc@ia - 1, Xcsc@ja - 1,
						 nnz, l1_reg, l2_reg, niter, nupd, step_size, 0, nthreads)
	} else {
		r_wrapper_poismf(A, B, dimA, dimB, k,
						 Xcsr@x, Xcsr@i, Xcsr@p,
						 Xcsc@x, Xcsc@i, Xcsc@p,
						 nnz, l1_reg, l2_reg, niter, nupd, step_size, 0, nthreads)
	}
	
	### Return all info
	A    <- matrix(A, nrow = k, ncol = dimA)
	B    <- matrix(B, nrow = k, ncol = dimB)
	Asum <- rowSums(A)
	Bsum <- rowSums(B)
	
	out <- list(
		A = A,
		B = B,
		Asum = Asum,
		Bsum = Bsum,
		k = k,
		l1_reg = l1_reg,
		l2_reg = l2_reg,
		niter = niter,
		nupd = nupd,
		step_size = step_size,
		init_type = init_type,
		dimA = dimA,
		dimB = dimB,
		nnz = nnz,
		nthreads = nthreads,
		seed = seed
	)
	if (is_non_int) {
		out[["levels_A"]] <- levels_A
		out[["levels_B"]] <- levels_B
	}
	return(structure(out, class = "poismf"))
}

#' @title Make predictions for arbitrary entries in matrix
#' @param object An object of class "poismf" as returned by function `poismf`.
#' @param a Row(s) for which to predict. Alternatively, a `data.frame` (first column being the column indices and
#' second column being the count values) or `sparseVector` (from package `Matrix`)
#' of counts for one row/user/document, from which predictions will be calculated by producing latent factors
#' on-the-fly.
#' @param b Column(s) for which to predict. If `NULL`, will make predictions for all columns. Otherwise,
#' it must be of the same length as "a", and the output will contain the prediction for each combination
#' of "a" and "b" passed here (unless passing `topN`).
#' @param seed Random seed to use to initialize factors (when `a` is a `data.frame` or `sparseVector`)
#' @param topN Return top-N ranked items (columns or IDs from "B") according to their predictions. If
#' passing argument "b", will return the top-N only among those.
#' @param l2_reg When passing to argument `a` a `data.frame` or `sparseVector` and the new factors needs to the calculated
#' on-the-fly, it indicates the L2 regularization strenght to use. Note that in this case, the new factors are optimized
#' through a conjugate-gradient routine, which works better with smaller regulatization values than the
#' proximal-gradient routine used to fit the model.
#' @param l1_reg L1 regularization to use in the same case as above.
#' @param ... Not used.
#' @seealso \link{poismf} \link{predict_all}
#' @export
#' @examples 
#' library(poismf)
#' 
#' ### create a random sparse data frame in COO format
#' nrow <- 10 ** 2
#' ncol <- 10 ** 3
#' nnz  <- 10 ** 4
#' set.seed(1)
#' X <- data.frame(
#'     row_ix = as.integer(runif(nnz, min = 1, max = nrow)),
#'     col_ix = as.integer(runif(nnz, min = 1, max = ncol)),
#'     count = rpois(nnz, 1) + 1)
#' X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
#' 
#' ### factorize the randomly-generated sparse matrix
#' model <- poismf(X, nthreads = 1)
#' 
#' ### predict functionality
#' predict(model, 1, 10) ## predict entry (1, 10)
#' predict(model, 1, topN = 10) ## predict top-10 entries "B" for row 1 of "A".
#' predict(model, c(1, 1, 1), c(4, 5, 6)) ## predict entries [1,4], [1,5], [1,6]
#' head(predict(model, 1)) ## predict the whole row 1
#' 
#' #all predictions for new row/user/doc
#' head(predict(model, data.frame(col_ix = c(1,2,3), count = c(4,5,6)) ))
predict.poismf <- function(object, a, b = NULL, seed = 10, topN = NULL, l2_reg = 1e3, l1_reg = 0, ...) {
	
	if (!is.null(topN) && ("numeric" %in% class(topN)))    { topN <- as.integer(topN) }
	if (!is.null(topN) && (!("integer" %in% class(topN)))) { stop("'topN' must be an integer.") }
	if (!is.null(topN)) {
		if ("numeric" %in% class(topN)) { topN <- as.integer(topN) }
		if (!("integer" %in% class(topN))) { stop("'topN' must be an integer.") }
		if (topN > object$dimB)            { stop("'topN' is larger than the number of rows in B.") }
		if (topN <= 0)                     { stop("'topN' must be a positive integer.") }
		if (!is.null(b) && topN > NROW(b)) { stop("'topN' is larger than vector 'b' that was passed.") }
	}
	class_a <- class(a)
	x_vec   <- NULL
	### check if factors need to be calculated on-the-fly
	if ("data.frame" %in% class_a) {
		x_vec <- as.numeric(a[[2]])
		a     <- as.integer(a[[1]])
	}
	if ("dsparseVector" %in% class_a) {
		x_vec <- as.numeric(a@x)
		a     <- as.integer(a@i)
	}
	
	if (!is.null(b) && length(b) > 1 && length(a) == 1) { a <- rep(a, length(b)) }
	if (!is.null(b) && is.null(x_vec) && length(a) != length(b)) { stop("'a' and 'b' must be of the same length.") }
	if ("levels_A" %in% names(object)) {
		if (is.null(x_vec)) {
			a <- as.integer(factor(a, levels = object$levels_A))
		} else {
			if ("dsparseVector" %in% class_a) {stop("Must provide column names as passed to function 'poismf' in 'X'.")}
			a <- as.integer(factor(a, levels = object$levels_B)) - 1
		}
		if (!is.null(b)) { b <- as.integer(factor(b, levels = object$levels_B)) }
	} else {
		a <- as.integer(a)
		if (min(a) < 1) { stop("'a' and 'b' must be row/column indexes.") }
		if (is.null(x_vec)) {
			if (max(a) > object$dimA) { stop("Can only make predictions for the same rows and columns from the training data.") }
		} else {
			a <- a - 1
			if (max(a) > object$dimB) { stop("Can only make predictions for the same columns from the training data.") }
		}
		if (!is.null(b)) {
			b <- as.integer(b)
			if (min(b) < 1) { stop("'a' and 'b' must be row/column indexes.") }
			if (max(b) > object$dimB) { stop("Can only make predictions for the same rows and columns from the training data.") }
		}
	}
	
	if (!is.null(x_vec)) {
		set.seed(seed)
		if (object$init_type == "gamma") {
			a_vec <- rgamma(object$k, 1)
		} else {
			a_vec <- runif(object$k)
		}
		factorize_single(a_vec, x_vec, a, length(a),
						 object$B, object$Bsum, object$k,
						 l1_reg, l2_reg)
	}
	
	if (is.null(b)) {
		if (is.null(x_vec)) {
			pred <- object$A[, a] %*% object$B
		} else {
			pred <- a_vec %*% object$B
		}
	} else {
		pred <- vector(mode = "numeric", length = length(b))
		if (is.null(x_vec)) {
			predict_multiple(object$A, object$B, object$k, length(b), a - 1, b - 1, pred, object$nthreads)
		} else {
			predict_multiple(a_vec, object$B, object$k, length(b), vector(mode="integer", length=length(b)),
							 b - 1, pred, object$nthreads)
		}
	}
	pred <- as.vector(pred)
	
	if (is.null(topN)) {
		if (!is.null(b) && "levels_B" %in% names(object)) {
			names(pred) <- object$levels_B
			return(pred)
		} else {
			return(pred)
		}
	} else {
		if (is.null(b)) { b <- 1:NROW(pred) }
		select_topN(pred, b, topN)
		b <- b[1:topN]
		if ("levels_B" %in% names(object)) {
			b <- object$levels_B[b]
		}
		return(b)
	}
}

#' @title Predict whole input matrix
#' @description Outputs the predictions for the whole input matrix to which the model was fit.
#' Note that this will be a dense matrix, and in typical recommender systems scenarios will
#' likely not fit in memory.
#' @param model A Poisson factorization model as output byfunction `poismf`.
#' @return A matrix A x B with the full predictions for all rows and column. If the inputs
#' did not have numbers as IDs, the equivalences to their IDs in the outputs are in the
#' `poismf` object under fields `levels_A` and `levels_B`.
#' @export
predict_all <- function(model) {
	if (class(model) != "poismf") {
		stop("'model' must be a 'poismf' object as produced by function 'poismf'.")
	}
	return(matrix(model$A, nrow = model$dimA, ncol = model$k) %*% t(matrix(model$B, nrow = model$dimB, ncol = model$k)))
}

#' @title Get information about poismf object
#' @description Print basic properties of a "poismf" object.
#' @param x An object of class "poismf" as returned by function "poismf".
#' @param ... Extra arguments (not used).
#' @export
print.poismf <- function(x, ...) {
	cat("Poisson Matrix Factorization\n\n")
	cat("Number of rows:", x$dimA, "\n")
	cat("Number of columns:", x$dimB, "\n")
	cat("Number of non-zero entries:", x$nnz, "\n")
	cat("Dimensionality of factorization:", x$k, "\n")
	cat("L1 regularization :", x$l1_reg, " - L2 regularization: ", x$l2_reg, "\n")
	cat("Iterations: ", x$niter, " - upd. per iter: ", x$nupd, "\n")
	cat("Initialization: ", x$init_type, " - Random seed: ", x$seed, "\n")
	
	if ("levels_A" %in% names(x)) {
		cat("\nRow names:", head(x$levels_A))
		cat("\nColumn names:", head(x$levels_B), "\n")
	}
}

#' @title Get information about poismf object
#' @description Print basic properties of a "poismf" object (same as `print.poismf` function).
#' @param object An object of class "poismf" as returned by function "poismf".
#' @param ... Extra arguments (not used).
#' @seealso \link{print.poismf}
#' @export
summary.poismf <- function(object, ...) {
	print.poismf(object)
}


factorize_single <- function(a_vector, x, ix, nnz, B, Bsum, k, l1_reg, l2_reg) {
	if (l1_reg > 0) { Bsum <- Bsum + l1_reg }
	nonneg.cg::minimize.nonneg.cg(calc_fun_single_R, calc_grad_single_R, a_vector,
								  tol=1e-3, maxnfeval=500, maxiter=200,
								  decr_lnsrch=.25, lnsrch_const=.01, max_ls=20,
								  extra_nonneg_tol=FALSE, nthreads=1, verbose=FALSE,
								  x, ix, nnz, B, Bsum, k, l2_reg, vector(mode = "numeric", length = k))
}

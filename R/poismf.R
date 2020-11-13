#' @importFrom methods as new
#' @importFrom utils head
#' @importFrom stats runif rpois
#' @importFrom Matrix t
#' @importClassesFrom Matrix dgCMatrix dgTMatrix dsparseVector
#' @useDynLib poismf, .registration=TRUE

#' @title Factorization of Sparse Counts Matrices through Poisson Likelihood
#' @description Creates a low-rank non-negative factorization of a sparse counts
#' matrix by maximizing Poisson likelihood minus L1/L2 regularization, using
#' gradient-based optimization procedures.
#' 
#' The model idea is: \eqn{\mathbf{X} \approx \texttt{Poisson}(\mathbf{A} \mathbf{B}^T)}{
#' X ~ Poisson(A*t(B))}
#' 
#' Ideal for usage in recommender systems, in which the `X` matrix would consist of
#' interactions (e.g. clicks, views, plays), with users representing the rows and items
#' representing the columns.
#' @param X The counts matrix to factorize. Can be: \itemize{
#' \item A `data.frame` with 3 columns, containing in this order:
#' row index or user ID, column index or item ID, count value. The first two columns will
#' be converted to factors to enumerate them internally, and will return those same
#' values from `topN`. In order to avoid this internal re-enumeration, can pass `X`
#' as a sparse COO matrix instead.
#' \item A sparse matrix from package `Matrix` in triplets (COO) format
#' (that is: `Matrix::dgTMatrix`) (recommended).
#' Such a matrix can be created from row/column indices through
#' function `Matrix::sparseMatrix` (with `giveCsparse=FALSE`).
#' Will also accept them in CSC format (`Matrix::dgCMatrix`), but will be converted
#' along the way (so it will be slightly slower).
#' \item A sparse matrix in COO format from the `SparseM` package. Such a matrix
#' can be created from row/column indices through
#' `new("matrix.coo", ra=values, ja=col_ix, ia=row_ix, dim=as.integer(c(m,n)))`.
#' Will also accept them in CSR and CSC format, but will be converted along the way
#' (so it will be slightly slower).
#' \item A full matrix (of class `base::matrix`) - this is not recommended though.
#' }
#' Passing sparse matrices is faster as it will not need to re-enumerate the rows and columns,
#' Full matrices will be converted to sparse.
#' @param k Number of latent factors to use (dimensionality of the low-rank factorization).
#' If `k` is small (e.g. `k=5`), it's recommended to use `method='pg'`.
#' For large values, (e.g. `k=100`), it's recommended to use `method='tncg'`.
#' For medium values (e.g. `k=50`), it's recommende to use either
#' `method='tncg'` or `method='cg'`.
#' @param method Optimization method to use. Options are: \itemize{
#' \item `"tncg"` : will use a truncated Newton-CG method. This is the
#' slowest option, but tends to find better local optima, and if run
#' for many iterations, tends to produce sparse latent factor
#' matrices.
#' \item `"cg"` : will use a Conjugate Gradient method, which is faster
#' than the truncated Newton-CG, but tends not to reach the best local
#' optima. Usually, with this method and the default hyperparameters,
#' the latent factor matrices will not be very sparse.
#' \item `"pg"` : will use a proximal gradient method, which is a lot faster
#' than the other two and more memory-efficient, but tends to only work
#' with very large regularization values, and doesn't find as good
#' local optima, nor tends to result in sparse factors.
#' }
#' @param limit_step When passing `method='cg'`, whether to limit the step sizes in each update
#' so as to drive at most one variable to zero each time, as prescribed in [2].
#' If running the procedure for many iterations, it's recommended to set this
#' to 'True'. You also might set `method='cg'` plus `maxupd=1` and
#' `limit_step=FALSE` to reduce the algorithm to simple gradient descent
#' with a line search.
#' @param l2_reg Strength of L2 regularization. It is recommended to use small values
#' along with `method='tncg'`, very large values along with `method='pg'`,
#' and medium to large values with `method='cg'`. If passing `"auto"`,
#' will set it to `10^3` for TNCG, `10^5` for CG, and `10^9` for PG.
#' @param l1_reg Strength of L1 regularization. Not recommended.
#' @param niter Number of alternating iterations to perform. One iteration denotes an update
#' over both matrices. If passing `'auto'`, will set it to 50 for TNCG,
#' 25 for CG, and 10 for PG.
#' @param maxupd Maximum number of updates to each user/item vector within an iteration.
#' Note: for 'method=TNCG', this means maximum number of \bold{function
#' evaluations} rather than number of updates, so it should be higher.
#' You might also want to try decreasing this while increasing `niter`.
#' For `method='pg'`, this will be taken as the actual number of updates,
#' as it does not perform a line search like the other methods.
#' If passing `"auto"`, will set it to `5*k` for `method='tncg'`,
#' 25 for `method='cg'`, and 10 for `method='pg'`. If using
#' `method='cg'`, you might also try instead setting `maxupd=1`
#' and `niter=100`.
#' @param initial_step Initial step size to use for proximal gradient updates. Larger step sizes
#' reach converge faster, but are more likely to result in failed optimization.
#' Ignored when passing `method='tncg'` or `method='cg'`, as those will
#' perform a line seach instead.
#' @param weight_mult Extra multiplier for the weight of the positive entries over the missing
#' entries in the matrix to factorize. Be aware that Poisson likelihood will
#' implicitly put more weight on the non-missing entries already. Passing larger
#' values will make the factors have larger values (which might be desirable),
#' and can help with instability and failed optimization cases. If passing this,
#' it's recommended to try very large values (e.g. 10^2), and might require
#' adjusting the other hyperparameters.
#' @param init_type How to initialize the model parameters. One of `'gamma'` (will initialize
#' them `~ Gamma(1, 1))` or `'unif'` (will initialize them `~ Unif(0, 1))`..
#' @param seed Random seed to use for starting the factorizing matrices.
#' @param handle_interrupt Whether to respond to interrupt signals in the optimization procedure.
#' If passing `TRUE`, whenever it receives an interrupt signal during the
#' optimzation procedure, it will termnate earlier, taking the current values
#' of the variables without finishing, instead of throwing an error.
#' If passing `FALSE`, will raise an error when it is interrupted, which
#' will only be catched after the procedure is finished, and the obtained
#' object will not be usable.
#' @param nthreads Number of parallel threads to use.
#' @details In order to obtain sparse latent factor matrices, you need to pass
#' `method='tncg'` and a large `niter`, such as `niter=50` or `niter=100`.
#' The L1 regularization approach is not recommended, even though it might
#' also produce relatively sparse results with the other optimization methods.
#' 
#' When using proximal gradient method, this model is prone to numerical
#' instability, and can turn out to spit all NaNs or zeros in the fitted
#' parameters. The conjugate gradient and Newton-CG methods are not prone to
#' such failed optimizations.
#' @references \itemize{
#' \item Cortes, David.
#' "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations."
#' arXiv preprint arXiv:1811.01908 (2018).
#' \item Li, Can.
#' "A conjugate gradient type method for the nonnegative constraints optimization problems."
#' Journal of Applied Mathematics 2013 (2013).
#' \item Carlsson, Christer, et al.
#' "User's guide for TN/TNBC: Fortran routines for nonlinear optimization."
#' Mathematical Sciences Dept. Tech. Rep. 307, The Johns Hopkins University. 1984.
#' }
#' @return An object of class `poismf` with the following fields of interest:
#' @field A The user/document/row-factor matrix (as a vector in row-major order,
#' has to be reshaped to (k, nrows) and then transposed to obtain an R matrix).
#' @field B The item/word/column-factor matrix (as a vector in row-major order,
#' has to be reshaped to (k, ncols) and then transposed to obtain an R matrix).
#' @field levels_A A vector indicating which user/row ID corresponds to each row
#' position in the `A` matrix. This will only be generated when passing `X` as a
#' `data.frame`, otherwise will not remap them.
#' @field levels_B A vector indicating which item/column ID corresponds to each row
#' position in the `B` matrix. This will only be generated when passing `X` as a
#' `data.frame`, otherwise will not remap them.
#' @export
#' @examples 
#' library(poismf)
#' 
#' ### create a random sparse data frame in COO format
#' nrow <- 10^2 ## <- users
#' ncol <- 10^3 ## <- items
#' nnz  <- 10^4 ## <- events (agg)
#' set.seed(1)
#' X <- data.frame(
#'         row_ix = sample(nrow, size=nnz, replace=TRUE),
#'         col_ix = sample(ncol, size=nnz, replace=TRUE),
#'         count  = rpois(nnz, 1) + 1
#'      )
#' X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
#' 
#' ### can also pass X as sparse matrix - see below
#' ### X <- Matrix::sparseMatrix(
#' ###          i=X$row_ix, j=X$col_ix, x=X$count,
#' ###          giveCsparse=FALSE)
#' ### the indices can also be characters or other types:
#' ### X$row_ix <- paste0("user", X$row_ix)
#' ### X$col_ix <- paste0("item", X$col_ix)
#' 
#' ### factorize the randomly-generated sparse matrix
#' ### good speed (proximal gradient)
#' model <- poismf(X, k=5, method="pg", nthreads=1)
#' 
#' ### good quality, but slower (conjugate gradient)
#' model <- poismf(X, k=5, method="cg", nthreads=1)
#' 
#' ### better quality, but much slower (truncated Newton-CG)
#' model <- poismf(X, k=5, method="tncg", nthreads=1)
#' 
#' \donttest{
#' ### for getting sparse factors
#' model <- poismf(X, k=50, method="tncg")
#' mean(model$A == 0.)
#' }
#' 
#' ### predict functionality (chosen entries in X)
#' ### predict entry [1, 10] (row 1, column 10)
#' predict(model, 1, 10)
#' ### predict entries [1,4], [1,5], [1,6]
#' predict(model, c(1, 1, 1), c(4, 5, 6))
#' 
#' ### ranking functionality (for recommender systems)
#' topN(model, user=2, n=5, exclude=X$col_ix[X$row_ix==2])
#' topN.new(model, X=X[X$row_ix==2, c("col_ix","count")],
#'     n=5, exclude=X$col_ix[X$row_ix==2])
#' 
#' ### obtaining latent factors
#' a_vec  <- factors.single(model,
#'             X[X$row_ix==2, c("col_ix","count")])
#' A_full <- factors(model, X)
#' A_orig <- get.factor.matrices(model)$A
#' 
#' ### (note that newly-obtained factors will differ slightly)
#' sqrt(mean((A_full["2",] - A_orig["2",])^2))
#' @seealso \link{predict.poismf} \link{topN} \link{factors}
#' \link{get.factor.matrices} \link{get.model.mappings}
poismf <- function(X, k = 50, method = "tncg",
                   l2_reg = "auto", l1_reg = 0,
                   niter = "auto", maxupd = "auto",
                   limit_step = TRUE, initial_step = 1e-7,
                   weight_mult = 1, init_type = "gamma", seed = 1,
                   handle_interrupt = TRUE,
                   nthreads = parallel::detectCores()) {
    
    ### Check input parameters
    allowed_methods <- c("tncg", "cg", "pg")
    if (!(method %in% allowed_methods) || (NROW(method) != 1))
        stop(paste0("'method' must be one of: ", paste(allowed_methods, collapse=", ")))
    if (NROW(k) > 1 || k < 1) { stop("'k' must be a positive integer.") }
    
    if (l2_reg == "auto")
        l2_reg <- switch(method, "tncg"=1e3, "cg"=1e5, "pg"=1e9)
    if (niter == "auto")
        niter <- switch(method, "tncg"=50L, "cg"=25L, "pg"=10L)
    if (maxupd == "auto")
        maxupd <- switch(method, "tncg"=5L*k, "cg"=5L, "pg"=1L)
    
    if (NROW(niter) > 1 || niter < 1) { stop("'niter' must be a positive integer.") }
    if (NROW(nthreads) > 1 || nthreads < 1) {nthreads <- parallel::detectCores()}
    if (NROW(limit_step) < 1) { stop("'limit_step' must be a boolean/logical.") }
    if (NROW(initial_step) > 1 || initial_step <= 0)
        stop("'initial_step' must be a positive number.")
    if (maxupd < 1) {stop("'maxupd' must be a positive integer.")}
    if (l1_reg < 0. | l2_reg < 0.) {stop("Regularization parameters must be non-negative.")}
    if (weight_mult <= 0.) { stop("'weight_mult' must be a positive number.") }
    if (init_type != "gamma" & init_type != "uniform")
        {stop("'init_type' must be one of 'gamma' or 'uniform'.")}
    
    ### Cast them to required type
    k            <- as.integer(k)
    l1_reg       <- as.numeric(l1_reg)
    l2_reg       <- as.numeric(l2_reg)
    method       <- as.character(method)
    limit_step   <- as.logical(limit_step)
    weight_mult  <- as.numeric(weight_mult)
    initial_step <- as.numeric(initial_step)
    niter        <- as.integer(niter)
    maxupd       <- as.integer(maxupd)
    nthreads     <- as.integer(nthreads)
    
    method_code  <- switch(method,
                           "tncg" = 1,
                           "cg"   = 2,
                           "pg"   = 3)
    method_code  <- as.integer(method_code)
    handle_interrupt <- as.logical(handle_interrupt)
    
    is_df <- FALSE
    
    ### Convert X to CSR and CSC
    if ("data.frame" %in% class(X)) {
        is_df    <- TRUE
        X[[1]]   <- factor(X[[1]])
        X[[2]]   <- factor(X[[2]])
        levels_A <- levels(X[[1]])
        levels_B <- levels(X[[2]])
        
        ix_row <- as.integer(X[[1]])
        ix_col <- as.integer(X[[2]])
        xflat  <- as.numeric(X[[3]])
        
        if (any(is.na(ix_row)) || any(is.na(ix_col)) || any(is.na(xflat))) {
            stop("Input contains missing values.")
        }
        if (any(ix_row < 1) | any(ix_col < 1)) {
            stop("First two columns of 'X' must be row/column indices starting at 1.")
        }
        ## Note: package 'Matrix' has only COO and CSC matrices; 'SparseM' has COO, CSR, CSC.
        ## SparseM will put the index under 'ja' and index pointer under 'ia', for both CSR and CSC.
        ## In order to create CSR matrices from 'Matrix', it makes CSC versions of their transpose.
        ## Also: 'Matrix' is zero-based, whereas 'SparseM' is one-based, except
        ## for vectors in 'Matrix' which are one-based; but when creating a CSC matrix, it will
        ## require the indices to be one-based and the indices pointers to be zero-based.
        Xcsr <- Matrix::sparseMatrix(i = ix_col, j = ix_row, x = xflat, giveCsparse = TRUE)
        Xcsc <- Matrix::sparseMatrix(i = ix_row, j = ix_col, x = xflat, giveCsparse = TRUE)
    } else if ("matrix" %in% class(X)) {
        Xcsr <- as(t(X), "dgCMatrix")
        Xcsc <- as(  X,  "dgCMatrix")
    } else if ("dgTMatrix" %in% class(X)) {
        Xcsr <- as(Matrix::t(X), "dgCMatrix")
        Xcsc <- as(X, "dgCMatrix")
    } else if ("dgCMatrix" %in% class(X)) {
        Xcsr <- Matrix::t(X)
        Xcsc <- X
    } else if ("matrix.coo" %in% class(X)) {
        if (requireNamespace("SparseM", quietly = TRUE)) {
            Xcsr <- SparseM::as.matrix.csr(X)
            Xcsc <- SparseM::as.matrix.csc(X)
        } else {
            Xcsr <- Matrix::sparseMatrix(i=X@ja, j=X@ia, x=X@ra,
                                         giveCsparse=TRUE, dims=rev(X@dimension))
            Xcsc <- Matrix::sparseMatrix(i=X@ia, j=X@ja, x=X@ra,
                                         giveCsparse=TRUE, dims=X@dimension)
        }
    } else if ("matrix.csr" %in% class(X)) {
        Xcsr <- X
        if (requireNamespace("SparseM", quietly = TRUE)) {
            Xcsc <- SparseM::t(X)
            class(Xcsc) <- "matrix.csc"
            Xcsc@dimension <- rev(Xcsc@dimension)
        } else {
            Xcoo <- Matrix::sparseMatrix(i=X@ja, p=X@ia-1L, x=X@ra,
                                         giveCsparse=FALSE, dims=rev(X@dimension))
            Xcsr <- as(Xcoo, "dgCMatrix")
            Xcsc <- as(Matrix::t(Xcoo), "dgCMatrix")
        }
    } else if ("matrix.csc" %in% class(X)) {
        Xcsc <- X
        if (requireNamespace("SparseM", quietly = TRUE)) {
            Xcsr <- SparseM::t(X)
            class(Xcsr) <- "matrix.csr"
            Xcsr@dimension <- rev(Xcsr@dimension)
        } else {
            Xcoo <- Matrix::sparseMatrix(i=X@ja, p=X@ia-1L, x=X@ra,
                                         giveCsparse=FALSE, dims=X@dimension)
            Xcsr <- as(Matrix::t(Xcoo), "dgCMatrix")
            Xcsc <- as(Xcoo, "dgCMatrix")
        }
    } else {
        stop("'X' must be a 'data.frame' with 3 columns, or a matrix (either full or sparse triplets).")
    }
    
    ### Get dimensions
    if ("matrix.csr" %in% class(Xcsr)) {
        nnz  <- length(Xcsr@ra)
        dimA <- NROW(Xcsr)
        dimB <- NCOL(Xcsr)
    } else {
        nnz  <- length(Xcsr@x)
        dimA <- NCOL(Xcsr)
        dimB <- NROW(Xcsr)
    }
    if (nnz < 1) { stop("Input does not contain non-zero values.") }
    
    ### Check for integer overflow
    if ((max(c(dimA, dimB, k)) > .Machine$integer.max) ||
        (min(c(dimA, dimB, k)) <= 0) ||
        any(c(is.na(dimA), is.na(dimB), is.na(k)))) {
        stop("Error: integer overflow. Dimensions cannot be larger than 2^31-1.")
    }
    size_within_int_range <- (
        .Call("check_size_below_int_max", dimA, k) &&
        .Call("check_size_below_int_max", dimB, k)
    )
    
    ### Initialize factor matrices
    set.seed(seed)
    if (size_within_int_range) {
        if (init_type == "gamma") {
            A <- -log(runif(dimA * k))
            B <- -log(runif(dimB * k))
        } else {
            A <- runif(dimA * k)
            B <- runif(dimB * k)
        }
    } else {
        A <- .Call("large_rnd_vec", dimA, k, init_type == "gamma")
        B <- .Call("large_rnd_vec", dimB, k, init_type == "gamma")
        if (NROW(A) == 0 || NROW(B == 0)) {
            stop("Memory error.")
        }
    }
    
    ### Run optimizer
    if ("matrix.csr" %in% class(Xcsr)) { ## 'SparseM'
        .Call("wrapper_run_poismf",
              Xcsr@ra, Xcsr@ja - 1L, Xcsr@ia - 1L,
              Xcsc@ra, Xcsc@ja - 1L, Xcsc@ia - 1L,
              A, B, dimA, dimB, k,
              method_code, limit_step, l2_reg, l1_reg,
              weight_mult, initial_step,
              niter, maxupd, handle_interrupt, nthreads)
    } else { ## 'Matrix'
        .Call("wrapper_run_poismf",
              Xcsr@x, Xcsr@i, Xcsr@p,
              Xcsc@x, Xcsc@i, Xcsc@p,
              A, B, dimA, dimB, k,
              method_code, limit_step, l2_reg, l1_reg,
              weight_mult, initial_step,
              niter, maxupd, handle_interrupt, nthreads)
    }
    
    ### Return all info
    Bsum <- rowSums(matrix(B, nrow = k, ncol = dimB)) + l1_reg
    
    out <- list(
        A = A,
        B = B,
        Bsum = Bsum,
        k = k,
        method = method,
        limit_step = limit_step,
        weight_mult = weight_mult,
        l1_reg = l1_reg,
        l2_reg = l2_reg,
        niter = niter,
        maxupd = maxupd,
        initial_step = initial_step,
        init_type = init_type,
        dimA = dimA,
        dimB = dimB,
        nnz = nnz,
        nthreads = nthreads,
        seed = seed
    )
    if (is_df) {
        out[["levels_A"]] <- levels_A
        out[["levels_B"]] <- levels_B
    }
    return(structure(out, class = "poismf"))
}

#' @title Poisson factorization with no input casting
#' @description This is a faster version of \link{poismf} which will not make any checks
#' or castings on its inputs. It is intended as a fast alternative when a model is to
#' be fit multiple times with different hyperparameters, and for allowing
#' custom-initialized factor matrices. \bold{Note that since it doesn't make any checks
#' or conversions, passing the wrong kinds of inputs or passing inputs with mismatching
#' dimensions will crash the R process}.
#' 
#' For most use cases, it's recommended to use the function `poismf` instead.
#' @param A Initial values for the user-factor matrix of dimensions [dimA, k],
#' assuming row-major order. Can be passed as a vector of dimension [dimA*k], or
#' as a matrix of dimension [k, dimA]. Note that R matrices use column-major order,
#' so if you want to pass an R matrix as initial values, you'll need to transpose it,
#' hence the shape [k, dimA]. Recommended to initialize `~ Gamma(1,1)`.
#' \bold{Will be modified in-place}.
#' @param B Initial values for the item-factor matrix of dimensions [dimB, k]. See
#' documentation about `A` for more details.
#' @param Xcsr The \bold{transpose} of the `X` matrix in CSC format (so that its structure
#' would match a CSR matrix). Should be an object of class `Matrix::dgCMatrix`.
#' @param Xcsc The `X` matrix in CSC format. Should be an object of class `Matrix::dgCMatrix`.
#' @param k The number of latent factors. \bold{Must match with the dimension of `A` and `B`}.
#' @param ... Other hyperparameters that can be passed to `poismf`. See the documentation
#' for \link{poismf} for details about possible hyperparameters. Init type and seed are
#' ignored as the `A` and `B` matrices are supposed to be passed already-initialized.
#' @return A `poismf` model object. See the documentation for \link{poismf} for details.
#' @examples
#' library(poismf)
#' 
#' ### create a random sparse data frame in COO format
#' nrow <- 10^2 ## <- users
#' ncol <- 10^3 ## <- items
#' nnz  <- 10^4 ## <- events (agg)
#' set.seed(1)
#' X <- data.frame(
#'         row_ix = sample(nrow, size=nnz, replace=TRUE),
#'         col_ix = sample(ncol, size=nnz, replace=TRUE),
#'         count  = rpois(nnz, 1) + 1
#'      )
#' X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
#' 
#' ### convert to required format
#' Xcsr <- Matrix::sparseMatrix(
#'             i=X$col_ix, j=X$row_ix, x=X$count,
#'             giveCsparse=TRUE
#' )
#' Xcsc <- Matrix::sparseMatrix(
#'             i=X$row_ix, j=X$col_ix, x=X$count,
#'             giveCsparse=TRUE
#' )
#' 
#' ### initialize factor matrices
#' k <- 5L
#' A <- rgamma(nrow*k, 1, 1)
#' B <- rgamma(ncol*k, 1, 1)
#' 
#' ### call function
#' model <- poismf_unsafe(A, B, Xcsr, Xcsc, k, nthreads=1)
#' @seealso \link{poismf}
#' @export
poismf_unsafe <- function(A, B, Xcsr, Xcsc, k, ...) {
    return(poismf__unsafe(A, B, Xcsr, Xcsc, k, ...))
}

poismf__unsafe <- function(A, B, Xcsr, Xcsc, k, method="tncg",
                           l2_reg="auto", l1_reg=0.,
                           niter="auto",
                           maxupd="auto",
                           limit_step=TRUE, initial_step=1e-7,
                           weight_mult=1.,
                           nthreads=parallel::detectCores(),
                           init_type=NULL,
                           seed=NULL, handle_interrupt=TRUE) {
    if (l2_reg == "auto")
        l2_reg <- switch(method, "tncg"=1e3, "cg"=1e5, "pg"=1e9)
    if (niter == "auto")
        niter <- switch(method, "tncg"=50L, "cg"=25L, "pg"=10L)
    if (maxupd == "auto")
        maxupd <- switch(method, "tncg"=5L*k, "cg"=5L, "pg"=1L)
    
    dimA <- NROW(A) / (ifelse(is.null(dim(A)), k, 1))
    dimB <- NROW(B) / (ifelse(is.null(dim(B)), k, 1))
    method_code <- switch(method,
                          "tncg" = 1,
                          "cg"   = 2,
                          "pg"   = 3)
    method_code <- as.integer(method_code)
    .Call("wrapper_run_poismf",
          Xcsr@x, Xcsr@i, Xcsr@p,
          Xcsc@x, Xcsc@i, Xcsc@p,
          A, B, dimA, dimB, k,
          method_code, limit_step, l2_reg, l1_reg,
          weight_mult, initial_step,
          niter, maxupd, handle_interrupt, nthreads)
    out <- list(
        A = A,
        B = B,
        Bsum = rowSums(matrix(B, nrow=k, ncol=dimB)) + l1_reg,
        k = k,
        method = method,
        limit_step = limit_step,
        weight_mult = weight_mult,
        l1_reg = l1_reg,
        l2_reg = l2_reg,
        niter = niter,
        maxupd = maxupd,
        initial_step = initial_step,
        init_type = "custom",
        dimA = dimA,
        dimB = dimB,
        nnz = NROW(Xcsr@x),
        nthreads = nthreads,
        seed = 0
    )
    class(out) <- "poismf"
    return(out)
}

#' @title Get latent factors for a new user given her item counts
#' @description This is similar to obtaining topics for a document in LDA. See also
#' function \link{factors} for getting factors for multiple users/rows at
#' a time.
#' 
#' This function works with one user at a time, and will use the
#' truncated Newton-CG approach regardless of how the model was fit.
#' Note that, since this optimization method will likely have
#' different optimal hyperparameters than the other methods, it
#' offers the option of varying those hyperparameters in here.
#' @details The factors are initialized to the mean of each column in the fitted model.
#' @param model Poisson factorization model as returned by `poismf`.
#' @param X Data with the non-zero item indices and counts for this new user. Can be
#' passed as a sparse vector from package `Matrix` (`Matrix::dsparseVector`, which can
#' be created from indices and values through function `Matrix::sparseVector`), or
#' as a `data.frame`, in which case will take the first column as the item/column indices
#' (numeration starting at 1) and the second column as the counts. If `X` passed to
#' `poismf` was a `data.frame`, `X` here must also be a `data.frame`.
#' @param l2_reg Strength of L2 regularization to use for optimizing the new factors.
#' @param l1_reg Strength of the L1 regularization. Not recommended.
#' @param weight_mult Weight multiplier for the positive entries over the missing entries.
#' @param maxupd Maximum number of Newton-CG updates to perform. You might want to
#' increase this value depending on the use-case.
#' @return Vector of dimensionality `model$k` with the latent factors for the user,
#' given the input data.
#' @seealso \link{factors} \link{topN.new}
#' @export
factors.single <- function(model, X, l2_reg = model$l2_reg, l1_reg = model$l1_reg,
                           weight_mult = model$weight_mult, maxupd = model$maxupd) {
    if ( ("levels_B" %in% names(model)) & !("data.frame" %in% class(X)) ) {
        stop("Must pass 'X' as data.frame if model was fit to X as data.frame.")
    }
    if (l2_reg < 0. | l1_reg < 0.) {
        stop("Regularization parameter must be positive.")
    }
    if (maxupd < 1) {
        stop("'maxupd' must be a positive integer.")
    }
    if (weight_mult <= 0.) {
        stop("'weight_mult' must be a positive number.")
    }
    
    maxupd <- as.integer(maxupd)
    l1_reg <- as.numeric(l1_reg)
    l2_reg <- as.numeric(l2_reg)
    weight_mult <- as.numeric(weight_mult)

    if ("data.frame" %in% class(X)) {
        xval <- as.numeric(X[[2]])
        xind <- process.items.vec(model, X[[1]], "First column of 'X'")
    } else if ("dsparseVector" %in% class(X)) {
        xval <- as.numeric(X@x)
        xind <- process.items.vec(model, X@i, "Column indices of 'X'")
    } else {
        stop("'X' must be a data.frame or Matrix::dsparseVector.")
    }
    
    return(.Call("wrapper_predict_factors",
                 model$A, model$k,
                 xval, xind,
                 model$B, model$Bsum,
                 maxupd, l2_reg,
                 l1_reg, model$l1_reg,
                 weight_mult))
}

#' @title Determine latent factors for new rows/users
#' @description Determines the latent factors for new users (rows) given their counts
#' for existing items (columns).
#' 
#' This function will use the same method and hyperparameters with which the
#' model was fit. If using this for recommender systems, it's recommended
#' to use instead the function \link{factors.single}, even though the new factors
#' obtained with that function might not end up being in the same scale as
#' the original factors in `model$A`.
#' 
#' Note that, when using proximal gradient method (the default), results from this function
#' and from `get.factor.matrices` on the same data might differ a lot. If this is a problem, it's
#' recommended to use conjugate gradient instead.
#' @details The factors are initialized to the mean of each column in the fitted model.
#' @param model A Poisson factorization model as returned by `poismf`.
#' @param X New data for whose rows to determine latent factors. Can be passed as
#' a `data.frame` or as a sparse or dense matrix (see documentation of \link{poismf}
#' for details on the data type). While other functions only accept sparse matrices
#' in COO (triplets) format, this function will also take CSR matrices from the
#' `SparseM` and  `Matrix` packages (classes `dgRMatrix`/`RsparseMatrix` for `Matrix`).
#' Inputs will be converted to CSR regardless of their original format.
#' 
#' Note that converting a matrix to `dgRMatrix` format might require using
#' `as(m, "RsparseMatrix")` instead of using `dgRMatrix` directly.
#' 
#' If passing a `data.frame`, the first column should contain row indices or IDs,
#' and these will be internally remapped - the mapping will be available as the row
#' names for the matrix if passing `add_names=TRUE`, or as part of the outputs if
#' passing `add_names=FALSE`. The IDs passed in the first column will not be matched
#' to the existing IDs of `X` passed to `poismf`.
#' 
#' If `X` passed to `poismf` was a `data.frame`, `X` here must also be passed as
#' `data.frame`. If `X` passed to `poismf` was a matrix and `X` is a `data.frame`,
#' the second column of `X` here should contain column numbers
#' (with numeration starting at 1).
#' @param add_names Whether to add row names to the output matrix if the indices
#' were internally remapped - they will only be so if the `X` here
#' is a `data.frame`. Note that if the indices in passed in `X` here (first and second
#' columns) are integers, once row names are added, subsetting `X` by an integer
#' will give the row at that position - that is, if you want to obtain the
#' corresponding row for ID=2 from `X` in `A_out`, you need to use `A_out["2", ]`,
#' not `A_out[2, ]`.
#' @return \itemize{
#'   \item If `X` was passed as a matrix, will output a matrix of dimensions (n, k)
#'   with the obtained factors. If passing `add_names=TRUE` and `X` passed to
#'   `poismf` was a `data.frame`, this matrix will have row names. \bold{Careful
#'   with subsetting with integers} (see documentation for `add_names`).
#'   \item If `X` was passed as a `data.frame` and passing `add_names=FALSE` here,
#'   will output a list with an entry `factors` containing the latent factors as
#'   described above, and an entry `mapping` indicating to which row ID does each
#'   row of the output correspond.
#' }
#' @seealso \link{factors.single} \link{topN.new}
#' @export
factors <- function(model, X, add_names=TRUE) {
    if ( ("levels_A" %in% class(model)) & !("data.frame" %in% class(X)) ) {
        stop("Must pass 'X' as data.frame if model was fit to X as data.frame.")
    }
    if ("data.frame" %in% class(X)) {
        fact <- factor(X[[1]])
        levs <- levels(fact)
        ixA  <- as.integer(fact)
        ixB  <- process.items.vec(model, X[[2]], "Second column of 'X_test'")
        Xval <- as.numeric(X[[3]])
        Xcsr <- Matrix::sparseMatrix(i=ixB+1L, j=ixA, x=Xval, giveCsparse=TRUE)
        dimA <- ncol(Xcsr)
    } else {
        levs <- NULL
        if (is.null(dim(X))) {
            stop("Invalid 'X'.")
        }
        if (ncol(X) != model$dimB)
            stop("'X' should contain the same columns as passed to 'poismf'.")
        
        if ("matrix" %in% class(X)) {
            Xcsr <- as(X, "dgRMatrix")
        } else if ("dgRMatrix" %in% class(X)) {
            Xcsr <- X
        } else if ("dgTMatrix" %in% class(X)) {
            Xcsr <- as(Matrix::t(X), "dgCMatrix")
        } else if("dgCMatrix" %in% class(X)) {
            Xcsr <- Matrix::t(X)
        } else if (NROW(intersect(c("matrix.coo", "matrix.csc"), class(X)))) {
            if (requireNamespace("SparseM", quietly = TRUE)) {
                Xcsr <- SparseM::as.matrix.csr(X)
            } else {
                if ("matrix.coo" %in% class(X)) {
                    Xcsr <- Matrix::sparseMatrix(i=X@ja, j=X@ia, x=X@ra,
                                                 giveCsparse=TRUE, dims=rev(X@dimension))
                } else {
                    Xcsr <- Matrix::t(Matrix::sparseMatrix(i=X@ja, p=X@ia-1L, x=X@ra,
                                                           giveCsparse=TRUE, dims=X@dimension))
                }
            }
        } else if ("matrix.csr" %in% class(X)) {
            Xcsr <- X
        } else {
            stop("'X' must be a 'data.frame' with 3 columns, or a matrix (either full, or sparse triplets or CSR).")
        }
        
        if (any(class(Xcsr) %in% c("dgRMatrix", "matrix.csr"))) {
            dimA <- nrow(Xcsr)
        } else {
            dimA <- ncol(Xcsr)
        }
        
    }
    
    if ("matrix.csr" %in% class(Xcsr)) {
        Xr_indptr  <- Xcsr@ia - 1L
        Xr_indices <- Xcsr@ja - 1L
        Xr_values  <- Xcsr@ra
    } else if ("dgRMatrix" %in% class(Xcsr)) {
        Xr_indptr  <- Xcsr@p
        Xr_indices <- Xcsr@j
        Xr_values  <- Xcsr@x
    } else {
        Xr_indptr  <- Xcsr@p
        Xr_indices <- Xcsr@i
        Xr_values  <- Xcsr@x
    }
    
    method_code <- switch(model$method,
                          "tncg" = 1,
                          "cg"   = 2,
                          "pg"   = 3)
    method_code <- as.integer(method_code)
    
    Anew <- .Call("wrapper_predict_factors_multiple",
                  model$A, as.integer(dimA), model$k,
                  model$B, model$Bsum,
                  Xr_indptr, Xr_indices, Xr_values,
                  model$l2_reg, model$weight_mult,
                  model$initial_step, model$niter, model$maxupd,
                  method_code, model$limit_step,
                  model$nthreads)
    if (is.null(Anew))
        stop("Memory error.")
    Anew <- t(matrix(Anew, nrow=model$k))
    if (add_names & ("levels_A" %in% names(model))) {
        row.names(Anew) <- levs
        levs <- NULL
    }
    if (is.null(levs)) {
        return(Anew)
    } else {
        return(list(factors=Anew, mapping=levs))
    }
}

#' @title Predict expected count for new row(user) and column(item) combinations
#' @param object A Poisson factorization model as returned by `poismf`.
#' @param a Can be either: \itemize{
#' \item A vector of length N with the users/rows to predict - each entry will be
#' matched to the corresponding entry at the same position in `b` - e.g. to predict
#' value for entries (3,4), (3,5), and (3,6), should pass `a=c(3,3,3), b=c(3,5,6)`.
#' If `X` passed to `poismf` was a `data.frame`, should match with the entries in
#' its first column. If `X` passed to `poismf` was a matrix, should indicate the
#' row numbers (numeration starting at 1).
#' \item A sparse matrix, ideally in COO (triplets) format from package `Matrix`
#' (`Matrix::dgTMatrix`) or from package `SparseM` (`matrix.coo`), in which case it
#' will make predictions for the non-zero entries in the matrix and will output
#' another sparse matrix with the predicted entries as values. In this case, `b`
#' should not be passed. This option is not available if the `X` passed to `poismf`
#' was a `data.frame`.
#' }
#' @param b A vector of length N with the items/columns to predict - each entry will be
#' matched to the corresponding entry at the same position in `a` - e.g. to predict
#' value for entries (3,4), (3,5), and (3,6), should pass `a=c(3,3,3), b=c(3,5,6)`.
#' If `X` passed to `poismf` was a `data.frame`, should match with the entries in
#' its second column. If `X` passed to `poismf` was a matrix, should indicate the
#' column numbers (numeration starting at 1). If `a` is a sparse matrix, should not
#' pass `b`.
#' @param ... Not used.
#' @return \itemize{
#' \item If `a` and `b` were passed, will return a vector of length N with the
#' predictions  for the requested row/column combinations.
#' \item If `b` was not passed, will return a sparse matrix with the same entries
#' and shape as `a`, but with the values being the predictions from the model for
#' the non-missing entries.
#' }
#' @seealso \link{poismf} \link{topN} \link{factors}
#' @export
predict.poismf <- function(object, a, b = NULL, ...) {
    if (is.null(a)) stop("Must pass 'a'.")
    
    if (is.null(b)) {
        if ("levels_A" %in% names(object)) {
            stop("Must pass 'b' when fitting the model was fit to a data.frame.")
        }
        
        outp_matrix <- TRUE
        return_csc  <- FALSE
        return_csr  <- FALSE
        if ("data.frame" %in% class(a)) {
            stop("Cannot pass a data.frame as 'a'.")
        }
        if ("matrix" %in% class(a)) {
            a <- as(a, "dgTMatrix")
        } else if ("dgCMatrix" %in% class(a)) {
            return_csc <- TRUE
            a <- as(a, "dgTMatrix")
        } else if (NROW(intersect(c("matrix.csr", "matrix.csc"), class(a)))) {
            if (requireNamespace("SparseM", quietly = TRUE)) {
                a <- SparseM::as.matrix.coo(a)
            } else {
                if ("matrix.csr" %in% class(a)) {
                    return_csr <- TRUE
                    a <- Matrix::t(Matrix::sparseMatrix(i=a@ja, p=a@ia-1L, x=a@ra,
                                                        giveCsparse=FALSE, dims=rev(a@dimension)))
                } else {
                    return_csc <- TRUE
                    a <- Matrix::sparseMatrix(i=a@ja, p=a@ia-1L, x=a@ra,
                                              giveCsparse=FALSE, dims=a@dimension)
                }
            }
        }
        
        if ("dgTMatrix" %in% class(a)) {
            ixA <- process.users.vec(object, a@i + 1L)
            ixB <- process.items.vec(object, a@j + 1L)
            mat_dims <- a@Dim
        } else if ("matrix.coo" %in% class(a)) {
            ixA <- process.users.vec(object, a@ia)
            ixB <- process.items.vec(object, a@ja)
            mat_dims <- a@dimension
        } else {
            stop("If not passing 'b', 'a' must be a sparse matrix.")
        }
    } else {
        outp_matrix <- FALSE
        ixA <- process.users.vec(object, a, "'a'")
        ixB <- process.items.vec(object, b, "'b'")
        if (length(ixA) != length(ixB)) {
            stop("'a' and 'b' must have the same number of entries.")
        }
    }
    
    pred <- .Call("wrapper_predict_multiple",
                  object$A, object$B, object$k,
                  ixA, ixB, object$nthreads)
    if (outp_matrix) {
        if (NROW(intersect(c("matrix", "dgTMatrix", "dgCMatrix"), class(a)))) {
            return(Matrix::sparseMatrix(i=ixA+1L, j=ixB+1L, x=pred,
                                        dims=mat_dims, giveCsparse=return_csc))
        } else {
            if (requireNamespace("SparseM", quietly = TRUE)) {
                out <- new("matrix.coo", ra=pred, ja=ixB+1L, ia=ixA+1L, dim=mat_dims)
                if (return_csr) {
                    return(SparseM::as.matrix.csr(out))
                } else if (return_csc) {
                    return(SparseM::as.matrix.csc(out))
                } else {
                    return(out)
                }
            } else {
                return(Matrix::sparseMatrix(i=ixA+1L, j=ixB+1L, x=pred,
                                            dims=mat_dims, giveCsparse=FALSE))
            }
        }
    } else {
        return(pred)
    }
}

process.users.vec <- function(model, users, errname) {
    if (is.null(users) || !NROW(users)) {
        return(integer(0))
    } else if ("levels_A" %in% names(model)) {
        users <- factor(users, model$levels_A)
    }
    
    users <- as.integer(users) - 1L
    umin <- min(users); umax <- max(users);
    if (is.null(users) || umin < 0 || umax >= model$dimA || any(is.na(users))) {
        stop(sprintf("%s contains invalid entries.", errname))
    }
    return(users)
}

process.items.vec <- function(model, items, errname) {
    if (is.null(items) || !NROW(items)) {
        return(integer(0))
    } else if ("levels_B" %in% names(model)) {
        items <- factor(items, model$levels_B)
    }
    
    items <- as.integer(items) - 1L
    imin <- min(items); imax <- max(items);
    if (is.null(items) || imin < 0 || imax >= model$dimB || any(is.na(items))) {
        stop(sprintf("%s contains invalid entries.", errname))
    }
    return(items)
}

topN_internal <- function(model, a_vec, n, include, exclude, output_score) {
    if (!is.null(include) & !is.null(exclude)) {
        stop("Can only pass one of 'include' or 'exclude'.")
    }
    if (NROW(n) != 1) stop("'n' must be a positive integer.")
    if (NROW(output_score) != 1) stop("'output_score' must be a single logical/boolean.")
    if (n > model$dimB) stop("'n' is larger than the available number of items.")
    
    n <- as.integer(n)
    output_score <- as.logical(output_score)
    include <- process.items.vec(model, include, "'include'")
    exclude <- process.items.vec(model, exclude, "'exclude'")
    if (NROW(include) > 0) {
        if (n < NROW(include)) stop("'n' cannot be smaller than the number of entries in 'include'.")
    }
    if (NROW(exclude) > 0) {
        if (n > (model$dimB - NROW(exclude))) stop("'n' is larger than the available number of items.")
    }
    
    outp_ix <- integer(n)
    outp_score <- numeric(0)
    if (output_score) outp_score <- numeric(n)
    .Call("wrapper_topN", outp_ix, outp_score,
          a_vec, model$B, model$dimB,
          include, exclude,
          n, model$nthreads)
    outp_ix <- outp_ix + 1
    if ("levels_B" %in% names(model)) {
        outp_ix <- model$levels_B[outp_ix]
    }
    if (output_score) {
        return(list(ix=outp_ix, score=outp_score))
    } else {
        return(outp_ix)
    }
}

#' @title Rank top-N highest-predicted items for an existing user
#' @param model A Poisson factorization model as returned by `poismf`.
#' @param user User for which to rank the items. If `X` passed to `poismf` was a
#' `data.frame`, must match with the entries in its first column,
#' otherwise should match with the rows of `X` (numeration starting at 1).
#' @param n Number of top-N highest-predicted results to output.
#' @param include List of items which will be ranked. If passing this, will only
#' make a ranking among these items. If `X` passed to `poismf` was a
#' `data.frame`, must match with the entries in its second column,
#' otherwise should match with the columns of `X` (numeration starting at 1). Can only pass
#' one of `include` or `exclude.` Must not contain duplicated entries.
#' @param exclude List of items to exclude from the ranking. If passing this, will
#' rank all the items except for these. If `X` passed to `poismf` was a
#' `data.frame`, must match with the entries in its second column,
#' otherwise should match with the columns of `X` (numeration starting at 1). Can only pass
#' one of `include` or `exclude`. Must not contain duplicated entries.
#' @param output_score Whether to output the scores in addition to the IDs. If passing
#' `FALSE`, will return a single array with the item IDs, otherwise
#' will return a list with the item IDs and the scores.
#' @return \itemize{
#'   \item If passing `output_score=FALSE` (the default), will return a vector of size `n`
#'   with the top-N highest predicted items for this user.If the `X` data passed to
#'   `poismf` was a `data.frame`, will contain the item IDs from its second column,
#'   otherwise will be integers matching to the columns of `X` (starting at 1). If
#'   `X` was passed as `data.frame`, the entries in this vector might be coerced to
#'   character regardless of their original type.
#'   \item If passing `output_score=TRUE`, will return a list, with the first entry
#'   being the vector described above under name `ix`, and the second entry being the
#'   associated scores, as a numeric vector of size `n`.
#' }
#' @seealso \link{topN.new} \link{predict.poismf} \link{factors.single}
#' @export
topN <- function(model, user, n = 10, include = NULL, exclude = NULL, output_score=FALSE) {
    if (NROW(user) != 1) stop("'user' must be a single ID or row number.")
    user <- process.users.vec(model, user, "'user'")
    a_vec <- model$A[(user*model$k + 1) : ((user+1)*model$k)]
    return(topN_internal(model, a_vec, n, include, exclude, output_score))
}

#' @title Rank top-N highest-predicted items for a new user
#' @details This function calculates the latent factors in the same way as
#' `factors.single` - see the documentation of \link{factors.single}
#' for details.
#' 
#' The factors are initialized to the mean of each column in the fitted model.
#' @param model A Poisson factorization model as returned by `poismf`.
#' @param X Data with the non-zero item indices and counts for this new user. Can be
#' passed as a sparse vector from package `Matrix` (`Matrix::dsparseVector`, which can
#' be created from indices and values through `Matrix::sparseVector`), or as a `data.frame`,
#' in which case will take the first column as the item/column indices
#' (numeration starting at 1) and the second column
#' as the counts. If `X` passed to `poismf` was a `data.frame`, `X` here must also be
#' a `data.frame`.
#' @param n Number of top-N highest-predicted results to output.
#' @param include List of items which will be ranked. If passing this, will only
#' make a ranking among these items. If `X` passed to `poismf` was a
#' `data.frame`, must match with the entries in its second column,
#' otherwise should match with the columns of `X` (numeration starting at 1). Can only pass
#' one of `include` or `exclude.` Must not contain duplicated entries.
#' @param exclude List of items to exclude from the ranking. If passing this, will
#' rank all the items except for these. If `X` passed to `poismf` was a
#' `data.frame`, must match with the entries in its second column,
#' otherwise should match with the columns of `X` (numeration starting at 1). Can only pass
#' one of `include` or `exclude`. Must not contain duplicated entries.
#' @param output_score Whether to output the scores in addition to the IDs. If passing
#' `FALSE`, will return a single array with the item IDs, otherwise
#' will return a list with the item IDs and the scores.
#' @param l2_reg Strength of L2 regularization to use for optimizing the new factors.
#' @param l1_reg Strength of the L1 regularization. Not recommended.
#' @param weight_mult Weight multiplier for the positive entries over the missing entries.
#' @param maxupd Maximum number of Newton-CG updates to perform. You might want to
#' increase this value depending on the use-case.
#' @return \itemize{
#'   \item If passing `output_score=FALSE` (the default), will return a vector of size `n`
#'   with the top-N highest predicted items for this user.If the `X` data passed to
#'   `poismf` was a `data.frame`, will contain the item IDs from its second column,
#'   otherwise will be integers matching to the columns of `X` (starting at 1). If
#'   `X` was passed as `data.frame`, the entries in this vector might be coerced to
#'   character regardless of their original type.
#'   \item If passing `output_score=TRUE`, will return a list, with the first entry
#'   being the vector described above under name `ix`, and the second entry being the
#'   associated scores, as a numeric vector of size `n`.
#' }
#' @seealso \link{factors.single} \link{topN}
#' @export
topN.new <- function(model, X, n = 10, include = NULL, exclude = NULL, output_score = FALSE,
                     l2_reg = model$l2_reg, l1_reg = model$l1_reg,
                     weight_mult = model$weight_mult, maxupd = model$maxupd) {
    a_vec <- factors.single(model, X, l2_reg=l2_reg, l1_reg=l1_reg,
                            weight_mult=weight_mult, maxupd=maxupd)
    return(topN_internal(model, a_vec, n, include, exclude, output_score))
}

#' @title Predict whole input matrix
#' @description Outputs the predictions for the whole input matrix to which the model was fit.
#' Note that this will be a dense matrix, and in typical recommender systems scenarios will
#' likely not fit in memory.
#' @param model A Poisson factorization model as output by function `poismf`.
#' @return A matrix [dimA x dimB] with the full predictions for all rows and columns.
#' If the `X` data passed to `poismf` was a `data.frame`, the resulting output will have
#' row and column names added to it. Be aware that, if `X` was a `data.frame` with integers
#' as IDs, selecting columns or rows of the named matrix will output ordinal positions
#' - e.g. `out[2,]` (second row) vs. `out["2",]` (row for ID=2 in `X`).
#' The mappings, if produced, can be obtained through function \link{get.model.mappings}.
#' @export
get.all.predictions <- function(model) {
    A <- t(matrix(model$A, nrow = model$k))
    B <-   matrix(model$B, nrow = model$k)
    
    out <- A %*% B
    if ("levels_A" %in% names(model))
        row.names(out) <- model$levels_A
    if ("levels_B" %in% names(model))
        colnames(out)  <- model$levels_B
    
    return(out)
}

#' @title Get information about poismf object
#' @description Print basic properties of a "poismf" object.
#' @param x An object of class "poismf" as returned by function "poismf".
#' @param ... Extra arguments (not used).
#' @export
print.poismf <- function(x, ...) {
    cat("Poisson Matrix Factorization\n\n")
    cat(sprintf("Method: %s\n", x$method))
    cat(sprintf("Number of rows: %d\n", x$dimA))
    cat(sprintf("Number of columns: %d\n", x$dimB))
    cat(sprintf("Number of non-zero entries: %d\n", x$nnz))
    cat(sprintf("Dimensionality of factorization: %d\n", x$k))
    cat(sprintf("L1 regularization :%g - L2 regularization: %g\n", x$l1_reg, x$l2_reg))
    cat(sprintf("Iterations: %d - max upd. per iter: %d\n", x$niter, x$maxupd))
    cat(sprintf("Initialization: %s", x$init_type))
    if (x$init_type != "custom") cat(sprintf(" - random seed: %d", x$seed))
    cat("\n")
    
    if ("levels_A" %in% names(x)) {
        cat("\nRow names:", head(x$levels_A), ifelse(NROW(x$levels_A) > 6, "...", ""))
        cat("\nCol names:", head(x$levels_B), ifelse(NROW(x$levels_B) > 6, "...", ""),
            "\n")
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

#' @title Extract Latent Factor Matrices
#' @description Extract the latent factor matrices for users (rows) and
#' columns (items) from a Poisson factorization model object, as returned
#' by function `poismf`.
#' @param model A Poisson factorization model, as produced by `poismf`.
#' @param add_names Whether to add row names to the matrices if the indices
#' were internally remapped - they will only be so if the `X` passed to `poismf`
#' was a `data.frame`. Note that if passing `X` as `data.frame` with integer indices
#' to `poismf`, once row names are added, subsetting such matrix by an integer will
#' give the row at that position - that is, if you want to obtain the corresponding
#' row for ID=2 from `X` in `factors$A`, you need to use `factors$A["2", ]`, not
#' `factors$A[2, ]`.
#' @return List with entries `A` (the user factors) and `B` (the item factors).
#' @details If `X` passed to `poismf` was a `data.frame`, the mapping between
#' IDs from `X` to row numbers in `A` and column numbers in `B` are avaiable under
#' `model$levels_A` and `model$levels_B`, respectively. They can also be obtained
#' through `get.model.mappings`, and will be added as row names if
#' using `add_names=TRUE`. \bold{Be careful about subsetting with integers} (see
#' documentation for `add_names` for details).
#' @seealso \link{get.model.mappings}
#' @export
get.factor.matrices <- function(model, add_names=TRUE) {
    A <- t(matrix(model$A, nrow = model$k))
    B <- t(matrix(model$B, nrow = model$k))
    if (add_names & ("levels_A" %in% names(model))) {
        row.names(A) <- model$levels_A
        row.names(B) <- model$levels_B
    }
    return(list(A = A, B = B))
}

#' @title Extract user/row and item/column mappings from Poisson model.
#' @description Will extract the mapping between IDs passed as `X` to
#' function `poismf` and row/column positions in the latent factor matrices
#' and prediction functions.
#' 
#' Such a mapping will only be generated if the `X` passed to `poismf` was a
#' `data.frame`, otherwise they will not be re-mapped.
#' @param model A Poisson factorization model as returned by `poismf`.
#' @return A list with row entries: \itemize{
#'   \item `rows`: a vector in which each user/row ID is placed at its ordinal position
#'   in the internal data structures. If there is no mapping (e.g. if `X` passed to
#'   `poismf` was a sparse matrix), will be `NULL`.
#'   \item `columns`: a vector in which each item/column ID is placed at its ordinal position
#'   in the internal data structures. If there is no mapping (e.g. if `X` passed to
#'   `poismf` was a sparse matrix), will be `NULL`.
#' }
#' @seealso \link{get.factor.matrices}
#' @export
get.model.mappings <-  function(model) {
    if (!("poismf" %in% class(model))) {
        stop("Must pass a 'poismf' model object.")
    }
    if ("levels_A" %in% names(model)) {
        return(list(rows = model$levels_A, columns = model$levels_B))
    } else {
        return(list(rows = NULL, columns = NULL))
    }
}

#' @title Evaluate Poisson log-likelihood for counts matrix
#' @description Calculates Poisson log-likehood plus constant for new combinations
#' of rows and columns of `X`. Intended to use as a test metric or for monitoring a validation set.
#' 
#' By default, this Poisson log-likelihood is calculated only for the combinations
#' of users (rows) and items (columns) provided in `X_test` here, ignoring the
#' missing entries. This is the usual use-case for evaluating a validation or test
#' set, but can also be used for evaluating it on the training data with all
#' missing entries included as zeros (see parameters for details).
#' 
#' Note that this calculates a \bold{sum} rather than an average.
#' @details If using more than 1 thread, the results might vary slightly between runs.
#' @param model A Poisson factorization model object as returned by `poismf`.
#' @param X_test Input data on which to calculate log-likelihood, consisting of triplets.
#' Can be passed as a `data.frame` or as a sparse COO matrix (see documentation of
#' \link{poismf} for details on the accepted data types). If the `X` data passed to
#' `poismf` was a `data.frame`, should pass a `data.frame` with entries corresponding
#' to the same IDs, otherwise might pass either a `data.frame` with the row and column
#' indices (starting at 1), or a sparse COO matrix.
#' @param full_llk Whether to add to the number a constant given by the data which doesn't
#' depend on the fitted parameters. If passing `False` (the default), there
#' is some chance that the resulting log-likelihood will end up being
#' positive - this is not an error, but is simply due to ommission of this
#' constant. Passing `TRUE` might result in numeric overflow and low
#' numerical precision.
#' @param include_missing If `TRUE`, will calculate the Poisson log-likelihood for all entries
#' (i.e. all combinations of users/items, whole matrix `X`),
#' taking the missing ones as zero-valued. If passing `FALSE`, will
#' calculate the Poisson log-likelihood only for the non-missing entries
#' passed in `X_test` - this is usually the desired behavior when
#' evaluating a test dataset.
#' @return Obtained Poisson log-likelihood (higher is better).
#' @seealso \link{poismf}
#' @export
poisson.llk <- function(model, X_test, full_llk = FALSE, include_missing = FALSE) {
    if ( ("levels_A" %in% class(model)) & !("data.frame" %in% class(X_test)) ) {
        stop("Must pass 'X_test' as data.frame if model was fit to X as data.frame.")
    }
    if ("data.frame" %in% class(X_test)) {
        ixA  <- process.users.vec(model, X_test[[1]], "First column of 'X_test'")
        ixB  <- process.items.vec(model, X_test[[2]], "Second column of 'X_test'")
        Xval <- as.numeric(X_test[[3]])
    } else {
        dimA <- nrow(X_test)
        dimB <- ncol(X_test)
        if (is.null(dimA) | is.null(dimB)) {
            stop("Invalid 'X_test'.")
        }
        if (dimA > model$dimA) stop("'X_test' cannot contain new rows.")
        if (dimB > model$dimB) stop("'X_test' cannot contain new columns.")
        
        if ("matrix" %in% class(X_test)) {
            X_test <- as(X_test, "dgTMatrix")
        }
        
        if ("dgCMatrix" %in% class(X_test)) {
            X_test <- as(X_test, "dgTMatrix")
        } else if ("matrix.csr" %in% class(X_test)) {
            if (requireNamespace("SparseM", quietly = TRUE)) {
                X_test <- SparseM::as.matrix.coo(X_test)
            } else {
                X_test <- Matrix::sparseMatrix(j=X_test@ja, p=X_test@ia-1L, x=X_test@ra,
                                               giveCsparse=FALSE)
            }
        } else if ("matrix.csc" %in% class(X_test)) {
            if (requireNamespace("SparseM", quietly = TRUE)) {
                X_test <- SparseM::as.matrix.coo(X_test)
            } else {
                X_test <- Matrix::sparseMatrix(i=X_test@ja, p=X_test@ia-1L, x=X_test@ra,
                                               giveCsparse=FALSE)
            }
        }
        
        if("dgTMatrix" %in% class(X_test)) {
            ixA  <- X_test@i
            ixB  <- X_test@j
            Xval <- X_test@x
        } else if ("matrix.coo" %in% class(X_test)) {
            ixA  <- X_test@ia - 1L
            ixB  <- X_test@ja - 1L
            Xval <- X_test@ra
        } else {
            stop("'X' must be a 'data.frame' with 3 columns, or a matrix (either full or sparse triplets).")
        }
        
    }
    
    return(.Call("wrapper_eval_llk",
                 model$A, model$B, model$dimA,
                 model$dimB, model$k,
                 ixA, ixB, Xval,
                 as.logical(full_llk), as.logical(include_missing),
                 model$nthreads))
}

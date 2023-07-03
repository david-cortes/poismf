import pandas as pd, numpy as np
import multiprocessing, os, warnings, ctypes
from scipy.sparse import issparse, coo_array
from copy import deepcopy
from . import c_funs_double, c_funs_float

__all__ = ["PoisMF"]

class PoisMF:
    """
    Poisson Matrix Factorization

    Fast and memory-efficient model for recommender systems based on Poisson
    factorization of sparse counts data (e.g. number of times a user played different
    songs), using gradient-based optimization procedures.

    The model idea is to approximate:
        :math:`\\mathbf{X} \\sim \\text{Poisson}(\\mathbf{A} \\mathbf{B}^T)`

    Note
    ----
    If passing ``reindex=True``, it will internally reindex all user and item IDs.
    Your data will not require reindexing if the IDs for users and items meet
    the following criteria:
        1) Are all integers.
        2) Start at zero.
        3) Don't have any enumeration gaps, i.e. if there is a user '4',
           user '3' must also be there.

    Note
    ----
    Although the main idea behind this software is to produce sparse model/factor
    matrices, they are always taken in dense format when used inside this software,
    and as such, it might be faster to use these matrices through some other external
    library that would be able to exploit their sparsity.

    Note
    ----
    When using proximal gradient method, this model is prone to numerical
    instability, and can turn out to spit all NaNs or zeros in the fitted
    parameters. The TNCG method is not prone to such failed optimizations.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank factorization).
        If ``k`` is very small (e.g. ``k=3``), it's recommended to use ``method='pg'``,
        otherwise it's recommended to use ``method='tncg'``, and if using ``method='cg'``,
        it's recommended to use large ``k`` (at least 100).
    method : bool
        Optimization method to use as inner solver. Options are:
            * ``"tncg"`` : will use the conjugate gradient method from reference [2]_.
              This is the slowest option, but tends to find better local optima, and
              if either run for many inner iterations (controlled by ``maxupd``) or
              reusing previous solutions each time (controlled by ``reuse_prev``),
              tends to produce sparse latent factor matrices.
              Note that when reusing previous solutions, fitting times are much faster
              and the quality of the results as evaluated by ranking-based recommendation
              quality metrics is almost as good, but solutions tend to be less sparse
              (see reference [1]_ for details).
              Unlike the other two, this solver is extremely unlikely to fail to produce
              results, and it is thus the recommended one.
            * ``"cg"`` : will use the conjugate gradient method from reference [3]_,
              which is faster than the one from reference [2]_, but tends not to reach
              as good local optima. Usually, with this method and the default hyperparameters,
              the latent factor matrices will be very sparse, but note that it can
              fail to produce results (in which case the obtained factors will be
              very poor quality without warning) when ``k`` is small (recommended to
              use ``k>=100`` when using this solver).
            * ``"pg"`` : will use a proximal gradient method, which is a lot faster
              than the other two and more memory-efficient, but tends to only work
              with very large regularization values, and doesn't find as good
              local optima, nor tends to result in sparse factors. Under this method,
              top-N recommendations tend to have little variation from one user to another.
    l2_reg : float
        Strength of L2 regularization. It is recommended to use small values
        along with ``method='tncg'``, very large values along with ``method='pg'``,
        and medium to large values with ``method='cg'``. If passing ``"auto"``,
        will set it to :math:`10^3` for TNCG, :math:`10^4` for CG, and :math:`10^9` for PG.
    l1_reg : float
        Strength of L1 regularization. Not recommended.
    niter : int
        Number of outer iterations to perform. One iteration denotes an update
        over both matrices. If passing ``'auto'``, will set it to 10 for TNCG and PG,
        or 30 for CG.

        Using more iterations usually leads to better results for CG, at the
        expense of longer fitting times. TNCG is more likely to converge to
        a local optimum with fewer outer iterations, with further iterations
        not changing the values of any single factor.
    maxupd : int
        Maximum number of inner iterations for each user/item vector within.
        **Note: for 'method=TNCG', this means maximum number of function
        evaluations rather than number of updates, so it should be higher.**
        You might also want to try decreasing this while increasing ``niter``.
        For ``method='pg'``, this will be taken as the actual number of updates,
        as it does not perform a line search like the other methods.
        If passing ``"auto"``, will set it to ``15*k`` for ``method='tncg'``,
        5 for ``method='cg'``, and 10 for ``method='pg'``. If using
        ``method='cg'``, one might also want to try other combinations such as
        ``maxupd=1`` and ``niter=100``.
    limit_step : bool
        When passing ``method='cg'``, whether to limit the step sizes in each update
        so as to drive at most one variable to zero each time, as prescribed in [2].
        If running the procedure for many iterations, it's recommended to set this
        to 'True'. You also might set ``method='cg'`` plus ``maxupd=1`` and
        ``limit_step=False`` to reduce the algorithm to simple projected gradient descent
        with a line search.
    initial_step : float
        Initial step size to use for proximal gradient updates. Larger step sizes
        reach converge faster, but are more likely to result in failed optimization.
        Ignored when passing ``method='tncg'`` or ``method='cg'``, as those will
        perform a line seach instead.
    early_stop : bool
        In the TNCG method, whether to stop before reaching the maximum number of
        iterations if the updates do not change the factors significantly or at all.
    reuse_prev : bool
        In the TNCG method, whether to reuse the factors obtained in the previous
        iteration as starting point for each inner update. This has the
        effect of reaching convergence much quicker, but will oftentimes lead to
        slightly worse solutions.

        If passing ``False`` and ``maxupd`` is small, the obtained factors might not
        be sparse at all. If passing ``True``, they will typically be less sparse
        than when passing ``False`` with large ``maxupd`` or than with ``method='cg'``.

        Setting it to ``True`` has the side effect of potentially making the factors
        obtained when fitting the model different from the factors obtained after
        calling the ``predict_factors`` function with the same data the model was fit.
        
        For methods other than TNCG, this is always assumed ``True``.
    weight_mult : float > 0
        Extra multiplier for the weight of the positive entries over the missing
        entries in the matrix to factorize. Be aware that Poisson likelihood will
        implicitly put more weight on the non-missing entries already. Passing larger
        values will make the factors have larger values (which might be desirable),
        and can help with instability and failed optimization cases. If passing this,
        it's recommended to try very large values (e.g. 10^2), and might require
        adjusting the other hyperparameters.
    random_state : int, RandomState, or Generator
        Random seed to use to initialize model parameters. If passing a NumPy
        'RandomState', will use it to draw a random integer as initial seed.
        If passing a NumPy 'Generator', will use it directly for drawing
        random numbers.
    reindex : bool
        Whether to reindex data internally. Will be ignored if passing a sparse
        COO array/matrix to 'fit'.
    copy_data : bool
        Whether to make deep copies of the data in order to avoid modifying it in-place.
        Passing 'False' is faster, but might modify the 'X' inputs in-place if they
        are DataFrames.
    produce_dicts : bool
        Whether to produce Python dictionaries for users and items, which
        are used to speed-up the prediction API of this package. You can still
        predict without them, but it might take some additional miliseconds
        (or more depending on the number of users and items).
    use_float : bool
        Whether to use C float type (typically ``np.float32``) instead of
        C double type (typically ``np.float64``). Using float types is
        faster and uses less memory, but it has less numerical precision,
        which is problematic with this type of model.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    nthreads : int
        Number of threads to use to parallelize computations.
        If passing a number lower than 1, will use the same formula as joblib
        does for calculating number of threads (which is
        n_cpus + 1 + n_jobs - i.e. pass -1 to use all available threads).
    n_jobs : int
        Synonym for 'nthreads'.
    
    Attributes
    ----------
    A : array (nusers, k)
        User-factor matrix.
    B : array (nitems, k)
        Item-factor matrix.
    user_mapping_ : array (nusers,)
        ID of the user (as passed to .fit) corresponding to each row of A.
    item_mapping_ : array (nitems,)
        ID of the item (as passed to .fit) corresponding to each row of B.
    user_dict_ : dict (nusers)
        Dictionary with the mapping between user IDs (as passed to .fit) and rows of A.
    item_dict_ : dict (nitems)
        Dictionary with the mapping between item IDs (as passed to .fit) and rows of B.
    is_fitted : bool
        Whether the model has been fit to some data.

    References
    ----------
    .. [1] Cortes, David.
           "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations."
           arXiv preprint arXiv:1811.01908 (2018).
    .. [2] Nash, Stephen G.
           "Newton-type minimization via the Lanczos method."
           SIAM Journal on Numerical Analysis 21.4 (1984): 770-788.
    .. [3] Li, Can.
           "A conjugate gradient type method for the nonnegative constraints optimization problems."
           Journal of Applied Mathematics 2013 (2013).
    """
    def __init__(self, k = 50, method = "tncg",
                 l2_reg = "auto", l1_reg = 0.0,
                 niter = "auto", maxupd = "auto",
                 limit_step = True, initial_step = 1e-7,
                 early_stop = True, reuse_prev = False,
                 weight_mult = 1.0, random_state = 1,
                 reindex = True, copy_data = True, produce_dicts = False,
                 use_float = True, handle_interrupt = True, nthreads = -1, n_jobs = None):
        self.k = k
        self.method = method
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.niter = niter
        self.maxupd = maxupd
        self.limit_step = limit_step
        self.initial_step = initial_step
        self.early_stop = early_stop
        self.reuse_prev = reuse_prev
        self.weight_mult = weight_mult
        self.random_state = random_state
        self.reindex = reindex
        self.copy_data = copy_data
        self.produce_dicts = produce_dicts
        self.use_float = use_float
        self.handle_interrupt = handle_interrupt
        self.nthreads = nthreads
        self.n_jobs = n_jobs

    def _init(self, k = 50, method = "tncg",
              l2_reg = "auto", l1_reg = 0.0,
              niter = "auto", maxupd = "auto",
              limit_step = True, initial_step = 1e-7,
              early_stop = True, reuse_prev = False,
              weight_mult = 1.0, random_state = 1,
              reindex = True, copy_data = True, produce_dicts = False,
              use_float = True, handle_interrupt = True, nthreads = -1, n_jobs = None):
        if n_jobs is not None:
            nthreads = n_jobs
        assert method in ["tncg", "cg", "pg"]
        if (isinstance(k, float) or
            isinstance(k, np.float32) or
            isinstance(k, np.float64)):
            k = int(k)

        ## default hyperparameters
        if l2_reg == "auto":
            l2_reg = {"tncg":1e3, "cg":1e4, "pg":1e9}[method]
        if maxupd == "auto":
            maxupd = {"tncg":15*k, "cg":5, "pg":10}[method]
        if niter == "auto":
            niter = {"tncg":10, "cg":30, "pg":10}[method]

        ## checking inputs
        assert k > 0
        assert isinstance(k, int) or isinstance(k, np.int64)
        assert niter >= 1
        assert maxupd >= 1
        assert isinstance(niter, int) or isinstance(niter, np.int64)
        assert isinstance(maxupd, int) or isinstance(maxupd, np.int64)
        assert l2_reg >= 0.
        assert l1_reg >= 0.
        assert initial_step > 0.
        assert weight_mult > 0.

        if isinstance(random_state, np.random.RandomState):
            random_state = random_state.randint(np.iinfo(np.int32).max)
        elif random_state is None:
            random_state = np.random.default_rng()
        elif isinstance(random_state, int) or isinstance(random_state, float):
            random_state = np.random.default_rng(seed=int(random_state))
        else:
            if not isinstance(random_state, np.random.Generator):
                raise ValueError("Invalid 'random_state'.")

        self._process_nthreads()
        
        ## storing these parameters
        self.k = k
        self.l1_reg_ = float(l1_reg)
        self.l2_reg_ = float(l2_reg)
        self.initial_step = float(initial_step)
        self.weight_mult = float(weight_mult)
        self.niter_ = niter
        self.maxupd_ = maxupd
        self.method = method
        self.limit_step = bool(limit_step)
        self.early_stop = bool(early_stop)
        self.reuse_prev = bool(reuse_prev)
        self.use_float = bool(use_float)
        self._dtype = ctypes.c_float if self.use_float else ctypes.c_double
        self.handle_interrupt = bool(handle_interrupt)

        self.reindex = bool(reindex)
        self.produce_dicts = bool(produce_dicts)
        if not self.reindex:
            self.produce_dicts = False
        self.random_state_ = random_state
        self.copy_data = bool(copy_data)
        
        self._reset_state()

    def _reset_state(self):
        self.A = np.empty((0,0), dtype=self._dtype)
        self.B = np.empty((0,0), dtype=self._dtype)
        self.user_mapping_ = np.empty(0, dtype=object)
        self.item_mapping_ = np.empty(0, dtype=object)
        self.user_dict_ = dict()
        self.item_dict_ = dict()
        self.is_fitted = False

    def _process_nthreads(self):
        if self.n_jobs is not None:
            nthreads = self.n_jobs
        else:
            nthreads = self.nthreads
        if nthreads < 1:
            nthreads = multiprocessing.cpu_count() + 1 + nthreads
        if nthreads is None:
            nthreads = 1
        assert nthreads > 0
        assert isinstance(nthreads, int) or isinstance(nthreads, np.int64)

        if (nthreads > 1) and not (c_funs_double._get_has_openmp()):
            msg_omp  = "Attempting to use more than 1 thread, but "
            msg_omp += "package was built without multi-threading "
            msg_omp += "support - see the project's GitHub page for "
            msg_omp += "more information."
            warnings.warn(msg_omp)

        self.nthreads_ = nthreads
    
    def fit(self, X):
        """
        Fit Poisson Model to sparse counts data

        Parameters
        ----------
        X : Pandas DataFrame (nobs, 3) or COO(m, n)
            Counts atrix to factorize, in sparse triplets format. Can be passed either
            as a SciPy sparse COO matrix (recommended) with users being rows and
            items being columns, or as a Pandas DataFrame, in which case it should
            have the following columns: 'UserId', 'ItemId', 'Count'.
            Combinations of users and items not present are implicitly assumed to be zero by the model. The non-missing entries must all be greater than zero.
            If passing a COO array/matrix, will force 'reindex' to 'False'.

        Returns
        -------
        self : obj
            This object
        """
        self._init(
            k = self.k, method = self.method,
            l2_reg = self.l2_reg, l1_reg = self.l1_reg,
            niter = self.niter, maxupd = self.maxupd,
            limit_step = self.limit_step, initial_step = self.initial_step,
            early_stop = self.early_stop, reuse_prev = self.reuse_prev,
            weight_mult = self.weight_mult, random_state = self.random_state,
            reindex = self.reindex, copy_data = self.copy_data, produce_dicts = self.produce_dicts,
            use_float = self.use_float, handle_interrupt = self.handle_interrupt,
            nthreads = self.nthreads, n_jobs = self.n_jobs
        )

        ## running each sub-process
        csr, csc = self._process_data(X)
        self._initialize_matrices()
        self._fit(csr, csc)
        self._produce_dicts()
        
        self.is_fitted = True
        return self
    
    def _process_data(self, X):
        if self.copy_data:
            X = X.copy()

        if issparse(X) and (X.format == "coo"):
            self.nusers = X.shape[0]
            self.nitems = X.shape[1]
            self.reindex = False
            coo = X

        elif isinstance(X, pd.DataFrame):
            cols_require = ["UserId", "ItemId", "Count"]
            cols_missing = np.setdiff1d(np.array(cols_require),
                                        X.columns.values)
            if cols_missing.shape[0]:
                raise ValueError("'X' should have columns: " + ", ".join(cols_require))
            
            if self.reindex:
                X['UserId'], self.user_mapping_ = pd.factorize(X.UserId)
                X['ItemId'], self.item_mapping_ = pd.factorize(X.ItemId)
                ### https://github.com/pandas-dev/pandas/issues/30618
                self.user_mapping_ = self.user_mapping_.to_numpy()
                self.item_mapping_ = self.item_mapping_.to_numpy()
                self.user_mapping_ = self.user_mapping_
                self.item_mapping_ = self.item_mapping_

            coo = coo_array((X.Count, (X.UserId, X.ItemId)))

        else:
            raise ValueError("'X' must be a pandas DataFrame or SciPy COO matrix.")

        self.nusers, self.nitems = coo.shape[0], coo.shape[1]
        csr = coo.tocsr()
        csc = coo.tocsc()

        csr.indptr  = csr.indptr.astype(ctypes.c_size_t)
        csr.indices = csr.indices.astype(ctypes.c_size_t)
        if csr.data.dtype != self._dtype:
            csr.data  = csr.data.astype(self._dtype)
        csc.indptr  = csc.indptr.astype(ctypes.c_size_t)
        csc.indices = csc.indices.astype(ctypes.c_size_t)
        if csc.data.dtype != self._dtype:
            csc.data  = csc.data.astype(self._dtype)

        return csr, csc
            

    def _initialize_matrices(self):
        ### This is the initialization that was used in the original HPF code
        self.A = 0.3 + self.random_state_.uniform(low=0, high=0.01, size=(self.nusers, self.k))
        self.B = 0.3 + self.random_state_.uniform(low=0, high=0.01, size=(self.nitems, self.k))
        if (self._dtype != self.A.dtype):
            self.A = self.A.astype(self._dtype)
            self.B = self.B.astype(self._dtype)
    
    def _fit(self, csr, csc):
        c_funs = c_funs_float if self.use_float else c_funs_double
        c_funs._run_poismf(
            csr.data, csr.indices, csr.indptr,
            csc.data, csc.indices, csc.indptr,
            self.A, self.B,
            self.method, self.limit_step,
            self.l2_reg_, self.l1_reg_, self.weight_mult,
            self.initial_step, self.niter_, self.maxupd_,
            self.early_stop, self.reuse_prev,
            self.handle_interrupt, self.nthreads_)
        self.Bsum = self.B.sum(axis = 0) + self.l1_reg_
        self.Amean = self.A.mean(axis = 0)

    def fit_unsafe(self, A, B, Xcsr, Xcsc):
        """
        Faster version for 'fit' with no input checks or castings

        This is intended as a faster alternative to ``fit`` when a model
        is to be fit multiple times with different hyperparameters. It will
        not make any checks or conversions on the inputs, as it will assume
        they are all in the right format.

        **Passing the wrong types of inputs or passing inputs with mismatching
        shapes will crash the Python process**. For most use cases, it's
        recommended to use ``fit`` instead.

        Note
        ----
        Calling this will override ``produce_dicts`` and ``reindex`` (will set
        both to ``False``).

        Parameters
        ----------
        A : array(dimA, k)
            The already-intialized user-factor matrices, as a NumPy array
            of type C dobule (this is usually ``np.float64``) in row-major
            order (a.k.a. C contiguous). Should **not** be a view/subset of a
            larger array (flag 'OWN_DATA'). Will be modified in-place.
        B : array(dimB, k)
            The already-initialized  item-factor matrices (see documentation
            about ``A`` for details on the format).
        Xcsr : CSR(dimA, dimB)
            The 'X' matrix to factorize in sparse CSR format (from SciPy).
            Must have the ``indices`` and ``indptr`` as type C size_t (this is
            usually ``np.uint64``). Note that setting them to this type might
            render the matrices unusable in SciPy's own sparse module functions.
            The ``data`` part must be of type  C double (this is usually
            ``np.float64``) or C float (usually ``np.float32``) depending
            on whether the object was constructed with ``use_float=True``
            or ``use_float=False``.
        Xcsc : CSC(dimA, dimB)
            The 'X' matrix to factorize in sparse CSC format (from SciPy).
            See documentation about ``Xcsr`` for details.
        
        Returns
        -------
        self : obj
            This object
        """
        self.A = A
        self.B = B
        self.nusers = A.shape[0]
        self.nitems = B.shape[0]
        self.reindex = False
        self.produce_dicts = False
        self._fit(Xcsr, Xcsc)
        self.is_fitted = True
        return self

    def _produce_dicts(self):
        if self.produce_dicts and self.reindex:
            self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
            self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}

    def predict_factors(self, X, l2_reg=None, l1_reg=None, weight_mult=None,
                        maxupd=None):
        """
        Get latent factors for a new user given her item counts

        This is similar to obtaining topics for a document in LDA. See also
        method 'transform' for getting factors for multiple users/rows at
        a time.

        Note
        ----
        This function works with one user at a time, and will use the
        TNCG solver regardless of how the model was fit.
        Note that, since this optimization method may have
        different optimal hyperparameters than the other methods, it
        offers the option of varying those hyperparameters in here.

        Note
        ----
        The factors are initialized to the mean of each column in the fitted model.

        Parameters
        ----------
        X : DataFrame or tuple(2)
            Either a DataFrame with columns 'ItemId' and 'Count', indicating the
            non-zero item counts for a user for whom it's desired to obtain
            latent factors, or a tuple with the first entry being the
            items/columns that have a non-zero count, and the second entry being
            the actual counts.
        l2_reg : float
            Strength of L2 regularization to use for optimizing the new factors.
            If passing ``None``, will take the value set in the model object.
        l1_reg : float
            Strength of the L1 regularization. Not recommended.
            If passing ``None``, will take the value set in the model object.
        weight_mult : float
            Weight multiplier for the positive entries over the missing entries.
            If passing ``None``, will take the value set in the model object.
        maxupd : int > 0
            Maximum number of TNCG updates to perform. You might want to
            increase this value depending on the use-case.
            If passing ``None``, will take the value set in the model object,
            clipped to a minimum of 1,000.

        Returns
        -------
        latent_factors : array (k,)
            Calculated latent factors for the user, given the input data
        """
        if l2_reg is None:
            l2_reg = self.l2_reg_
        if l1_reg is None:
            l1_reg = self.l1_reg_
        if weight_mult is None:
            weight_mult = self.weight_mult
        if maxupd is None:
            maxupd = max(1000, self.maxupd_)

        assert weight_mult > 0.
        ix, cnt = self._process_data_single(X)
        l2_reg, l1_reg = self._process_reg_params(l2_reg, l1_reg)
        c_funs = c_funs_float if self.use_float else c_funs_double
        a_vec = c_funs._predict_factors(cnt, ix,
                                        self.B, self.Bsum,
                                        self.Amean,
                                        self.reuse_prev,
                                        int(maxupd),
                                        float(l2_reg),
                                        float(l1_reg), float(self.l1_reg_),
                                        float(weight_mult))
        if np.any(np.isnan(a_vec)):
            raise ValueError("NaNs encountered in the result. Failed to produce latent factors.")
        if np.max(a_vec) <= 0:
            raise ValueError("Optimization failed. Could not calculate latent factors.")
        return a_vec

    def _process_data_single(self, X):
        assert self.is_fitted

        if self.copy_data:
            if isinstance(X, tuple):
                X = deepcopy(X)
            else:
                X = X.copy()

        if isinstance(X, pd.DataFrame):
            assert X.shape[0] > 0
            assert 'ItemId' in X.columns.values
            assert 'Count' in X.columns.values
            X = [X.ItemId.to_numpy(), X.Count.to_numpy()]
        elif isinstance(X, tuple) or isinstance(X, list):
            X = [np.array(X[0]), np.array(X[1])]
            if X[0].shape[0] != X[1].shape[0]:
                raise ValueError("'X' must have the same number of entries for items and counts.")
        else:
            raise ValueError("'X' must be a DataFrame or tuple.")

        if self.reindex:
            X[0] = pd.Categorical(X[0], self.item_mapping_).codes
        imin, imax = X[0].min(), X[0].max()
        if (imin < 0) or (imax >= self.nitems):
            raise ValueError("'X' contains invalid items.")

        if X[0].dtype != ctypes.c_size_t:
            X[0] = X[0].astype(ctypes.c_size_t)
        if X[1].dtype != self._dtype:
            X[1] = X[1].astype(self._dtype)

        return X[0], X[1]

    def _process_reg_params(self, l2_reg, l1_reg):
        if isinstance(l2_reg, int) or  isinstance(l2_reg, np.int64):
            l2_reg = float(l2_reg)
        if isinstance(l1_reg, int) or isinstance(l1_reg, np.int64):
            l1_reg = float(l1_reg)
        assert isinstance(l1_reg, float)
        assert isinstance(l2_reg, float)
        return l2_reg, l1_reg

    def transform(self, X, y=None):
        """
        Determine latent factors for new rows/users

        Note
        ----
        This function will use the same method and hyperparameters with which the
        model was fit. If using this for recommender systems, it's recommended
        to use instead the function 'predict_factors' as it's likely to be more precise.

        Note
        ----
        When using ``method='pg'`` (not recommended), results from this function
        and from 'fit' on the same datamight differ a lot.

        Note
        ----
        This function is prone to producing all zeros or all NaNs values.

        Note
        ----
        The factors are initialized to the mean of each column in the fitted model.
        
        Parameters
        ----------
        X : DataFrame(nnz, 3), CSR(n_new, nitems), or COO(n_new, nitems)
            New matrix for which to determine latent factors. The items/columns
            must be the same ones as passed to 'fit', while the rows correspond
            to new entries that were not passed to 'fit'.
                * If passing a DataFrame, must have columns 'UserId', 'ItemId', 'Count'.
                  The 'UserId' column will be remapped, with the mapping returned as
                  part of the output.
                * If passing a COO array/matrix, will be casted to CSR.
                * If passing a CSR array/matrix (recommended), will use it as-is and will
                  not return a mapping as the output will match row-by-row with 'X'.
        y : None
            Ignored. Kept in place for compatibility with SciKit-Learn pipelining.

        Returns
        -------
        A_new : array(n_new, k)
            The obtained latent factors for each row of 'X'.
        user_mapping : array(n_new)
            The mapping from 'UserId' in 'X' to rows of 'A_new'. Only
            produced when passing 'X' as a DataFrame, in which case the
            output will be a tuple '(A_new, user_mapping_)'.
        """
        assert self.is_fitted
        assert X.shape[0] > 0
        self._process_nthreads()
        Xr_indptr, Xr_indices, Xr, user_mapping_ = self._process_X_new_users(X)
        c_funs = c_funs_float if self.use_float else c_funs_double
        A = c_funs._predict_factors_multiple(
                self.B,
                self.Bsum,
                self.Amean,
                Xr_indptr,
                Xr_indices,
                Xr,
                self.l2_reg_,
                self.weight_mult,
                self.initial_step,
                self.niter_,
                self.maxupd_,
                self.method,
                self.limit_step,
                self.reuse_prev,
                self.nthreads_
        )

        if user_mapping_.shape[0]:
            return A, user_mapping_
        else:
            return A

    def _process_X_new_users(self, X):
        if self.copy_data:
            X = X.copy()

        if isinstance(X, pd.DataFrame):
            cols_require = ["UserId", "ItemId", "Count"]
            if np.setdiff1d(np.array(cols_require), X.columns.values).shape[0]:
                raise ValueError("'X' must contain columns " + ", ".join(cols_require))
            X["UserId"], user_mapping_ = pd.factorize(X.UserId)
            if self.reindex:
                X["ItemId"] = pd.Categorical(X.ItemId, self.item_mapping_).codes
            X = coo_array((X.Count, (X.UserId, X.ItemId))).tocsr()
        else:
            if self.reindex:
                raise ValueError("'X' must be a DataFrame if using 'reindex=True'.")
            if issparse(X) and (X.format != "csr"):
                X = X.tocsr()
            else:
                raise ValueError("'X' must be a DataFrame, CSR matrix, or COO matrix.")
            user_mapping_ = np.empty(0, dtype=int)

        if X.shape[1] > self.nitems:
            raise ValueError("'X' must have the same columns (items) as passed to 'fit'.")

        return (
            X.indptr.astype(ctypes.c_size_t),
            X.indices.astype(ctypes.c_size_t),
            X.data.astype(self._dtype),
            user_mapping_
        )

    
    def predict(self, user, item):
        """
        Predict expected count for combinations of users (rows) and items (columns)
        
        Note
        ----
        You can either pass an individual user and item, or arrays representing
        tuples (UserId, ItemId) with the combinatinons of users and items for which
        to predict (one entry per prediction).

        Parameters
        ----------
        user : array-like (npred,) or obj
            User(s) for which to predict each item.
        item: array-like (npred,) or obj
            Item(s) for which to predict for each user. Each entry will
            be matched with the corresponding entry in ``user``.
        
        Returns
        -------
        pred : array(npred,)
            Expected counts for the requested user(row)/item(column) combinations.
        """
        assert self.is_fitted
        self._process_nthreads()
        if isinstance(user, list) or isinstance(user, tuple):
            user = np.array(user)
        if isinstance(item, list) or isinstance(item, tuple):
            item = np.array(item)
        if isinstance(user, pd.Series):
            user = user.to_numpy()
        if isinstance(item, pd.Series):
            item = item.to_numpy()
            
        if isinstance(user, np.ndarray):
            if len(user.shape) > 1:
                user = user.reshape(-1)
            assert user.shape[0] > 0
            if self.reindex:
                if user.shape[0] > 1:
                    user = pd.Categorical(user, self.user_mapping_).codes
                else:
                    if len(self.user_dict_):
                        try:
                            user = self.user_dict_[user]
                        except Exception:
                            user = -1
                    else:
                        user = pd.Categorical(user, self.user_mapping_).codes[0]
        else:
            if self.reindex:
                if len(self.user_dict_):
                    try:
                        user = self.user_dict_[user]
                    except Exception:
                        user = -1
                else:
                    user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
            user = np.array([user])
            
        if isinstance(item, np.ndarray):
            if len(item.shape) > 1:
                item = item.reshape(-1)
            assert item.shape[0] > 0
            if self.reindex:
                if item.shape[0] > 1:
                    item = pd.Categorical(item, self.item_mapping_).codes
                else:
                    if len(self.item_dict_):
                        try:
                            item = self.item_dict_[item]
                        except Exception:
                            item = -1
                    else:
                        item = pd.Categorical(item, self.item_mapping_).codes[0]
        else:
            if self.reindex:
                if len(self.item_dict_):
                    try:
                        item = self.item_dict_[item]
                    except Exception:
                        item = -1
                else:
                    item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
            item = np.array([item])

        assert user.shape[0] == item.shape[0]
        
        if user.shape[0] == 1:
            if (user[0] < 0) or (item[0] < 0) or (user[0] >= self.nusers) or (item[0] >= self.nitems):
                return np.nan
            else:
                return self.A[user].dot(self.B[item].T).reshape(-1)[0]
        else:
            c_funs = c_funs_float if self.use_float else c_funs_double
            nan_entries = (user  < 0) | (item < 0) | (user >= self.nusers) | (item >= self.nitems)
            if np.any(nan_entries):
                if user.dtype != ctypes.c_size_t:
                    user = user.astype(ctypes.c_size_t)
                if item.dtype != ctypes.c_size_t:
                    item = item.astype(ctypes.c_size_t)
                out = np.empty(user.shape[0], dtype = self._dtype)
                c_funs._predict_multiple(out, self.A, self.B, user,
                                         item, self.nthreads_)
                return out
            else:
                non_na_user = user[~nan_entries]
                non_na_item = item[~nan_entries]
                out = np.empty(user.shape[0], dtype = self._dtype)
                temp = np.empty(np.sum(~nan_entries), dtype = self._dtype)
                c_funs._predict_multiple(temp, self.A, self.B,
                                         non_na_user.astype(ctypes.c_size_t),
                                         non_na_item.astype(ctypes.c_size_t),
                                         self.nthreads_)
                out[~nan_entries] = temp
                out[nan_entries] = np.nan
                return out
        
    
    def topN(self, user, n=10, include=None, exclude=None, output_score=False):
        """
        Rank top-N highest-predicted items for an existing user

        Note
        ----
        Even though the fitted model matrices might be sparse, they are always used
        in dense format here. In many cases it might be more efficient to produce the
        rankings externally through some library that would exploit the sparseness for
        much faster computations. The matrices can be access under ``self.A`` and ``self.B``.


        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'UserId' column,
            otherwise should match with the rows of 'X'.
        n : int
            Number of top-N highest-predicted results to output.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include' or 'exclude'. Must not contain duplicated entries.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include' or 'exclude'. Must not contain duplicated entries.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            'fit' was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        assert isinstance(n, int) or isinstance(n, np.int64)
        assert n >= 1
        if n > self.nitems:
            raise ValueError("'n' is larger than the available number of items.")
        if user is None:
            raise ValueError("Must pass a valid user.")
        self._process_nthreads()


        if self.reindex:
            if len(self.user_dict_):
                user = self.user_dict_[user]
            else:
                user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
                if user < 0:
                    raise ValueError("Invalid 'user'.")
        else:
            assert isinstance(user, int) or isinstance(user, np.int64)
            if (user < 0) or (user > self.nusers):
                raise ValueError("Invalid 'user'.")

        include, exclude = self._process_include_exclude(include, exclude)
        if include.shape[0]:
            if n < include.shape[0]:
                raise ValueError("'n' cannot be smaller than the number of entries in 'include'.")
        if exclude.shape[0]:
            if n > (self.nitems - exclude.shape[0]):
                raise ValueError("'n' is larger than the number of available items.")

        c_funs = c_funs_float if self.use_float else c_funs_double
        outp_ix, outp_score = c_funs._call_topN(
            self.A[user],
            self.B,
            include,
            exclude,
            n,
            output_score,
            self.nthreads_
        )

        if self.reindex:
            outp_ix = self.item_mapping_[outp_ix]
        if output_score:
            return outp_ix, outp_score
        else:
            return outp_ix


    def _process_include_exclude(self, include, exclude):
        if (include is not None) and (exclude is not None):
            raise ValueError("Can only pass one of 'include' or 'exclude'.")

        if include is not None:
            if isinstance(include, list) or isinstance(include, tuple):
                include = np.array(include)
            elif isinstance(include, pd.Series):
                include = include.to_numpy()
            elif not isinstance(include, np.ndarray):
                raise ValueError("'include' must be a list, tuple, Series, or array.")

        if exclude is not None:
            if isinstance(exclude, list) or isinstance(exclude, tuple):
                exclude = np.array(exclude)
            elif isinstance(exclude, pd.Series):
                exclude = exclude.to_numpy()
            elif not isinstance(exclude, np.ndarray):
                raise ValueError("'exclude' must be a list, tuple, Series, or array.")


        if self.reindex:
            if (include is not None):
                if len(self.item_dict_):
                    include = np.array([self.item_mapping_[i] for i in include])
                else:
                    include = pd.Categorical(include, self.item_mapping_).codes
            if (exclude is not None):
                if len(self.item_dict_):
                    exclude = np.array([self.item_mapping_[i] for i in exclude])
                else:
                    exclude = pd.Categorical(exclude, self.item_mapping_).codes
        
        if include is not None:
            imin, imax = include.min(), include.max()
            if (imin < 0) or (imax >= self.nitems):
                raise ValueError("'include' contains invalid entries.")
        else:
            include = np.empty(0, dtype=ctypes.c_size_t)

        if exclude is not None:
            imin, imax = exclude.min(), exclude.max()
            if (imin < 0) or (imax >= self.nitems):
                raise ValueError("'exclude' contains invalid entries.")
        else:
            exclude = np.empty(0, dtype=ctypes.c_size_t)

        if include.dtype != ctypes.c_size_t:
            include = include.astype(ctypes.c_size_t)
        if exclude.dtype != ctypes.c_size_t:
            exclude = exclude.astype(ctypes.c_size_t)
        return include, exclude


    def topN_new(self, X, n=10, include=None, exclude=None, output_score=False,
                 l2_reg = None, l1_reg = None, weight_mult=1.,
                 maxupd = None):
        """
        Rank top-N highest-predicted items for a new user

        Note
        ----
        This function calculates the latent factors in the same way as
        ``predict_factors`` - see the documentation of ``predict_factors``
        for details.

        Just like ``topN``, it does not exploit any potential sparsity in the
        fitted matrices and vectors, so it might be a lot faster to produce the
        recommendations externally (see the documentation for ``topN`` for details).

        Note
        ----
        The factors are initialized to the mean of each column in the fitted model.

        Parameters
        ----------
        X : DataFrame or tuple(2)
            Either a DataFrame with columns 'ItemId' and 'Count', indicating the
            non-zero item counts for a user for whom it's desired to obtain latent
            factors, or a tuple with the first entry being the items/columns that
            have a non-zero count, and the second entry being the actual counts.
        n : int
            Number of top-N highest-predicted results to output.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include' or 'exclude'. Must not contain duplicated entries.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include' or 'exclude'. Must not contain duplicated entries.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.
        l2_reg : float
            Strength of L2 regularization to use for optimizing the new factors.
            If passing ``None``, will take the value set in the model object.
        l1_reg : float
            Strength of the L1 regularization. Not recommended.
            If passing ``None``, will take the value set in the model object.
        weight_mult : float
            Weight multiplier for the positive entries over the missing entries.
            If passing ``None``, will take the value set in the model object.
        maxupd : int > 0
            Maximum number of TNCG updates to perform. You might want to
            increase this value depending on the use-case.
            If passing ``None``, will take the value set in the model object,
            clipped to a minimum of 1,000.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            'fit' was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec = self.predict_factors(X, l2_reg=l2_reg, l1_reg=l1_reg,
                                     weight_mult=weight_mult, maxupd=maxupd)
        include, exclude = self._process_include_exclude(include, exclude)
        if include.shape[0]:
            if n < include.shape[0]:
                raise ValueError("'n' cannot be smaller than the number of entries in 'include'.")
        if exclude.shape[0]:
            if n > (self.nitems - exclude.shape[0]):
                raise ValueError("'n' is larger than the number of available items.")
        self._process_nthreads()

        c_funs = c_funs_float if self.use_float else c_funs_double
        outp_ix, outp_score = c_funs._call_topN(
            a_vec,
            self.B,
            include,
            exclude,
            n,
            output_score,
            self.nthreads_
        )

        if self.reindex:
            outp_ix = self.item_mapping_[outp_ix]
        if output_score:
            return outp_ix, outp_score
        else:
            return outp_ix

    def _process_new_data(self, X_test):
        assert self.is_fitted

        if self.copy_data:
            X_test = X_test.copy()

        if self.reindex:
            if isinstance(X_test, pd.DataFrame):
                raise ValueError("If using 'reindex=True', 'X_test' must be a DataFrame.")
            cols_require = ["UserId", "ItemId", "Count"]
            if np.setdiff1d(np.array(cols_require), X_test.columns.values).shape[0]:
                raise ValueError("'X_test' must contain columns " + ", ".join(cols_require))
            X_test["UserId"] = pd.Categorical(X_test.UserId, self.user_mapping_).codes
            X_test["ItemId"] = pd.Categorical(X_test.UserId, self.item_mapping_).codes
            if X_test.UserId.dtype != ctypes.c_size_t:
                X_test["UserId"] = X_test["UserId"].astype(ctypes.c_size_t)
            if X_test.ItemId.dtype != ctypes.c_size_t:
                X_test["ItemId"] = X_test["ItemId"].astype(ctypes.c_size_t)
            if X_test.Count.dtype != self._dtype:
                X_test["Count"] = X_test["Count"].astype(self._dtype)

            umin, umax = X_test.UserId.min(), X_test.UserId.max()
            imin, imax = X_test.ItemId.min(), X_test.ItemId.max()
            if (umin < 0) or (umax > self.nusers):
                raise ValueError("'X_test' contains invalid users.")
            if (imin < 0) or (imax > self.nitems):
                raise ValueError("'X_test' contains invalid items.")

            return (
                X_test.UserId.to_numpy(),
                X_test.ItemId.to_numpy(),
                X_test.Count.to_numpy()
            )
        else:
            if isinstance(X_test, pd.DataFrame):
                X_test = coo_array((X_test.Count, (X_test.UserId, X_test.ItemId)))
            else:
                if not (issparse(X_test) and (X_test.format == "coo")):
                    raise ValueError("'X_test' must be a DataFrame or COO matrix.")

        if X_test.shape[0] > self.nusers:
            raise ValueError("'X_test' cannot contain new users/rows.")
        if X_test.shape[1] > self.nitems:
            raise ValueError("'X_test' cannot contain new items/columns.")

        return (
            X_test.row.astype(ctypes.c_size_t),
            X_test.col.astype(ctypes.c_size_t),
            X_test.data.astype(self._dtype)
        )

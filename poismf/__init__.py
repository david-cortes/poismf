import pandas as pd, numpy as np
import multiprocessing, os, warnings, ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from .poismf_c_wrapper import run_pgd, _predict_multiple, _predict_factors
pd.options.mode.chained_assignment = None

class PoisMF:
    """
    Poisson Matrix Factorization

    Fast and memory-efficient odel for recommending items based on Poisson factorization
    on sparse count data (e.g. number of times a user played different songs),
    using either proximal or conjugate gradient optimization procedures.
    
    If passing reindex=True, it will internally reindex all user and item IDs. Your data will not require
    reindexing if the IDs for users and items in counts_df meet the following criteria:

    1) Are all integers.
    2) Start at zero.
    3) Don't have any enumeration gaps, i.e. if there is a user '4', user '3' must also be there.

    If you only want to obtain the fitted parameters and use your own API later for recommendations,
    you can pass produce_dicts=False and pass a folder where to save them in csv format (they are also
    available as numpy arrays in this object's A and B attributes). Otherwise, the model
    will create Python dictionaries with entries for each user and item, which can take quite a bit of
    RAM memory. These can speed up predictions later through this package's API.

    Note
    ----
    DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
    need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

    Note
    ----
    This model is prone to numerical instability and can turn out to spit all NaNs or zeros in the fitted
    parameters.

    Parameters
    ----------
    k : int
        Number of latent factors to use.
    l2_reg : float
        Strength of L2 regularization. Recommended to decrease for conjugate gradient.
    l1_reg : float
        Strength of L1 regularization. Not recommended.
    niter : int
        Number of alternating iterations to perform (will perform two swaps per iteration).
    npasses : int
        Number of proximal  gradient updates to perform to each vector per iteration. Increasing the number
        of iterations has the same computational complexity and is likely to produce better results. Ignored
        for conjugate gradient method.
    initial_step : float
        Initial step size to use (ignored for conjugate gradient method).
    use_cg : bool
        Whether to fit the model through conjugate gradient method (slower, but less prone to failure).
    init_type : str
        How to initialize the model parameters. One of 'gamma' (will initialize them ~ Gamma(1, 1))
        or 'unif' (will initialize them ~ Unif(0, 1)).
    random_seed : int
        Random seed to use to initialize model parameters.
    nthreads : int
        Number of threads to use to parallelize computations.
        If set to 0 or less, will use the maximum available on the computer.
    reindex : bool
        Whether to reindex data internally.
    keep_data : bool
        Whether to keep information about which user was associated with each item
        in the training set, so as to exclude those items later when making Top-N
        recommendations.
    save_folder : str or None
        Folder where to save all model parameters as csv files.
    produce_dicts : bool
        Whether to produce Python dictionaries for users and items, which
        are used to speed-up the prediction API of this package. You can still predict without
        them, but it might take some additional miliseconds (or more depending on the
        number of users and items).
    
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
    [1] Cortes, David. "Fast Non-Bayesian Poisson Factorization for Implicit-Feedback Recommendations." arXiv preprint arXiv:1811.01908 (2018).
    """
    def __init__(self, k = 40, l2_reg = 1e9, l1_reg = 0.0, niter = 10, npasses = 1, initial_step = 1e-7,
                 use_cg = False, init_type = 'gamma', random_seed = 1, nthreads = -1,
                 reindex=True, keep_data = True, save_folder = None, produce_dicts = True):

        ## checking input
        assert k > 0
        assert isinstance(k, int)
        assert niter >= 1
        assert npasses >= 1
        assert isinstance(niter, int)
        assert isinstance(npasses, int)
        assert l2_reg >= 0
        assert l1_reg >= 0
        assert initial_step > 0
        assert isinstance(l2_reg, float)
        assert isinstance(l1_reg, float)
        assert isinstance(initial_step, float)
        assert init_type in ['gamma', 'unif']
        
        if nthreads < 1:
            nthreads = multiprocessing.cpu_count()
        if nthreads is None:
            nthreads = 1
        assert nthreads > 0
        assert isinstance(nthreads, int)

        if random_seed is not None:
            assert isinstance(random_seed, int)
        else:
            random_seed = np.random.randint(int(1e5)) + 1
        assert random_seed > 0
        assert isinstance(random_seed, int)
            
        if save_folder is not None:
            save_folder = os.path.expanduser(save_folder)
            assert os.path.exists(save_folder)
        
        ## storing these parameters
        self.k = k
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.initial_step = initial_step
        self.random_seed = random_seed
        self.init_type = init_type
        self.niter = niter
        self.npasses = npasses
        self.use_cg = int(bool(use_cg))
        self.nthreads = nthreads

        self.reindex = bool(reindex)
        self.keep_data = bool(keep_data)
        self.save_folder = save_folder
        self.produce_dicts = bool(produce_dicts)
        if not self.reindex:
            self.produce_dicts = False
        
        ## initializing other attributes
        self.A = None
        self.B = None
        self.user_mapping_ = None
        self.item_mapping_ = None
        self.user_dict_ = None
        self.item_dict_ = None
        self.is_fitted = False
    
    def fit(self, counts_df):
        """
        Fit Poisson Model to sparse count data
        
        Note
        ----
        DataFrames and arrays passed to '.fit' might be modified inplace - if this is a problem you'll
        need to pass a copy to them, e.g. 'counts_df=counts_df.copy()'.

        Parameters
        ----------
        counts_df : pandas data frame (nobs, 3) or coo_matrix
            Input data with one row per non-zero observation, consisting of triplets ('UserId', 'ItemId', 'Count').
            Must containin columns 'UserId', 'ItemId', and 'Count'.
            Combinations of users and items not present are implicitly assumed to be zero by the model.
            Can also pass a sparse coo_matrix, in which case 'reindex' will be forced to 'False'.

        Returns
        -------
        self : obj
            This object
        """

        ## running each sub-process
        self._process_data(counts_df)
        self._write_parameters()
        self._initialize_matrices()
        self._fit()
        
        ## after terminating optimization
        if self.keep_data:
            self._store_metadata()
        if self.produce_dicts and self.reindex:
            self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
            self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}
        self.is_fitted = True
        del self._csr
        del self._csc
        
        return self

    def _write_parameters(self):
        if self.save_folder is not None:
            with open(os.path.join(self.save_folder, "hyperparameters.txt"), "w") as pf:
                pf.write("l1_reg: %.10f\n" % self.l1_reg)
                pf.write("l2_reg: %.10f\n" % self.l2_reg)
                pf.write("initial_step: %.10f\n" % self.initial_step)
                pf.write("use_cg: %s\n" % self.use_cg)
                pf.write("niter: %d\n" % self.niter)
                pf.write("npasses: %d\n" % self.npasses)
                pf.write("k: %d\n" % self.k)
                if self.random_seed is not None:
                    pf.write("random seed: %d\n" % self.random_seed)
                else:
                    pf.write("random seed: None\n")
    
    def _process_data(self, input_df):
        calc_n = True
        is_coo = False

        if isinstance(input_df, np.ndarray):
            assert len(input_df.shape) > 1
            assert input_df.shape[1] >= 3
            self.input_df = pd.DataFrame(input_df[:, :3])
            self.input_df.columns = ['UserId', 'ItemId', "Count"]
            
        elif input_df.__class__.__name__ == 'DataFrame':
            assert input_df.shape[0] > 0
            assert 'UserId' in input_df.columns.values
            assert 'ItemId' in input_df.columns.values
            assert 'Count'  in input_df.columns.values
            self.input_df = input_df[['UserId', 'ItemId', 'Count']]
            
        elif input_df.__class__.__name__ == 'coo_matrix':
            self.nusers = input_df.shape[0]
            self.nitems = input_df.shape[1]
            self._coo = self.input_df
            self.reindex = False
            calc_n = False
            is_coo = True
        else:
            raise ValueError("'input_df' must be a pandas data frame, numpy array, or scipy sparse coo_matrix.")

        if is_coo:
            obs_zero = self._coo.data <= 0
        else:
            obs_zero = self.input_df.Count.values <= 0
        if obs_zero.sum() > 0:
            msg = "'counts_df' contains observations with a count value less than 1, these will be ignored."
            msg += " Any user or item associated exclusively with zero-value observations will be excluded."
            msg += " If using 'reindex=False', make sure that your data still meets the necessary criteria."
            msg += " If you still want to use these observations, set 'stop_crit' to 'diff-norm' or 'maxiter'."
            warnings.warn(msg)
            if is_coo:
                self._coo = coo_matrix((self._coo.data[~obs_zero], (self._coo.row[~obs_zero], self._coo.col[~obs_zero])))
                self.nusers = input_df.shape[0]
                self.nitems = input_df.shape[1]
            else:
                self.input_df = self.input_df.loc[~obs_zero]
            
        if self.reindex:
            self.input_df['UserId'], self.user_mapping_ = pd.factorize(self.input_df.UserId)
            self.input_df['ItemId'], self.item_mapping_ = pd.factorize(self.input_df.ItemId)
            self.nusers = self.user_mapping_.shape[0]
            self.nitems = self.item_mapping_.shape[0]
            self.user_mapping_ = np.array(self.user_mapping_).reshape(-1)
            self.item_mapping_ = np.array(self.item_mapping_).reshape(-1)
            if (self.save_folder is not None) and self.reindex:
                pd.Series(self.user_mapping_).to_csv(os.path.join(self.save_folder, 'users.csv'), index=False)
                pd.Series(self.item_mapping_).to_csv(os.path.join(self.save_folder, 'items.csv'), index=False)
        else:
            if calc_n:
                self.nusers = self.input_df.UserId.max() + 1
                self.nitems = self.input_df.ItemId.max() + 1

        if not is_coo:
            self._coo = coo_matrix((self.input_df.Count, (self.input_df.UserId, self.input_df.ItemId)))
            del self.input_df

        self._csr = csr_matrix(self._coo)
        self._csc = csc_matrix(self._coo)
        del self._coo

        self._csr.indptr  = self._csr.indptr.astype(ctypes.c_size_t)
        self._csr.indices = self._csr.indices.astype(ctypes.c_size_t)
        self._csr.data    = self._csr.data.astype(ctypes.c_double)
        self._csc.indptr  = self._csc.indptr.astype(ctypes.c_size_t)
        self._csc.indices = self._csc.indices.astype(ctypes.c_size_t)
        self._csc.data    = self._csc.data.astype(ctypes.c_double)

        return None
            
    def _store_metadata(self):
        ### https://github.com/numpy/numpy/issues/8333
        self._n_seen_by_user = (self._csr.indptr[1:] - self._csr.indptr[:-1]).astype(int)
        self._st_ix_user = self._csr.indptr[:-1].astype(int)
        self._seen = self._csr.indices.astype(int)

    def _initialize_matrices(self):
        np.random.seed(self.random_seed)
        if self.init_type == "gamma":
            self.A = np.random.gamma(1, 1, size = (self.nusers, self.k))
            self.B = np.random.gamma(1, 1, size = (self.nitems, self.k))
        else:
            self.A = np.random.random(size = (self.nusers, self.k))
            self.B = np.random.random(size = (self.nitems, self.k))
    
    def _fit(self):
        run_pgd(
            self._csr.data, self._csr.indices, self._csr.indptr,
            self._csc.data, self._csc.indices, self._csc.indptr,
            self.A, self.B,
            self.use_cg, self.l2_reg, self.l1_reg,
            self.initial_step, self.niter, self.npasses, self.nthreads)
        self.Bsum = self.B.sum(axis = 0).reshape(-1).astype(ctypes.c_double) + self.l1_reg

    def _process_data_single(self, counts_df):
        assert self.is_fitted
        if isinstance(counts_df, np.ndarray):
            assert len(counts_df.shape) > 1
            assert counts_df.shape[1] >= 2
            counts_df = counts_df.values[:,:2]
            counts_df.columns = ['ItemId', "Count"]
            
        if counts_df.__class__.__name__ == 'DataFrame':
            assert counts_df.shape[0] > 0
            assert 'ItemId' in counts_df.columns.values
            assert 'Count' in counts_df.columns.values
            counts_df = counts_df[['ItemId', 'Count']]
        else:
            raise ValueError("'counts_df' must be a pandas data frame or a numpy array")
            
        if self.reindex:
            if self.produce_dicts:
                try:
                    counts_df['ItemId'] = counts_df.ItemId.map(lambda x: self.item_dict_[x])
                except:
                    raise ValueError("Can only make calculations for items that were in the training set.")
            else:
                counts_df['ItemId'] = pd.Categorical(counts_df.ItemId.values, self.item_mapping_).codes
                if (counts_df.ItemId == -1).sum() > 0:
                    raise ValueError("Can only make calculations for items that were in the training set.")

        counts_df['ItemId'] = counts_df.ItemId.values.astype(ctypes.c_size_t)
        counts_df['Count']  = counts_df.Count.values.astype(ctypes.c_double)
        return counts_df

    def predict_factors(self, counts_df, random_seed=1):
        """
        Gets latent factors for a user given her item counts

        This is similar to obtaining topics for a document in LDA.

        Note
        ----
        This function will NOT modify any of the item parameters.

        Note
        ----
        This function only works with one user at a time.

        Note
        ----
        This function is prone to producing all zeros or all NaNs values.

        Parameters
        ----------
        counts_df : DataFrame or array (nsamples, 2)
            Data Frame with columns 'ItemId' and 'Count', indicating the non-zero item counts for a
            user for whom it's desired to obtain latent factors.
        random_seed : int
            Random seed used to initialize parameters.

        Returns
        -------
        latent_factors : array (k,)
            Calculated latent factors for the user, given the input data
        """
        return self._predict_factors(counts_df, random_seed, False)

    def _predict_factors(self, counts_df, random_seed=1, return_counts=False):
        if random_seed is None:
            random_seed = np.random.randint(int(1e5)) + 1
        assert isinstance(random_seed, int)
        assert random_seed > 0

        ## processing the data
        counts_df = self._process_data_single(counts_df)

        ## calculating the latent factors
        if self.init_type == "gamma":
            a_vec = np.random.gamma(1, 1, size = self.k)
        else:
            a_vec = np.random.random(size = self.k)
        _predict_factors(a_vec, counts_df.Count.values, counts_df.ItemId.values,
                         self.B, self.Bsum, self.l2_reg)

        if np.any(np.isnan(a_vec)):
            raise ValueError("NaNs encountered in the result. Failed to produce latent factors.")
        if np.max(a_vec) <= 0:
            raise ValueError("Optimization failed. Could not calculate latent factors.")

        if return_counts:
            return a_vec, counts_df
        else:
            return a_vec

    def add_user(self, user_id, counts_df, update_existing=False, random_seed=1):
        """
        Add a new user to the model or update parameters for a user according to new data
        
        Note
        ----
        This function will NOT modify any of the item parameters.

        Note
        ----
        This function can be run in parallel (through Python's joblib with shared memory) if using 'keep_data = False'

        Note
        ----
        This function is prone to producing all-zeros factors. For betters results, refit the model again from scratch.

        Parameters
        ----------
        user_id : obj
            Id to give to be user (when adding a new one) or Id of the existing user whose parameters are to be
            updated according to the data in 'counts_df'. **Make sure that the data type is the same that was passed
            in the training data, so if you have integer IDs, don't pass a string as ID**.
        counts_df : data frame or array (nsamples, 2)
            Data Frame with columns 'ItemId' and 'Count'. If passing a numpy array, will take the first two columns
            in that order. Data containing user/item interactions **from one user only** for which to add or update
            parameters. Note that you need to pass *all* the user-item interactions for this user when making an update,
            not just the new ones.
        update_existing : bool
            Whether this should be an update of the parameters for an existing user (when passing True), or
            an addition of a new user that was not in the model before (when passing False).
        random_seed : int
            Random seed used to initialize parameters.

        Returns
        -------
        True : bool
            Will return True if the process finishes successfully.
        """
        if update_existing:
            ## checking that the user already exists
            if self.produce_dicts and self.reindex:
                user_id = self.user_dict_[user_id]
            else:
                if self.reindex:
                    user_id = pd.Categorical(np.array([user_id]), self.user_mapping_).codes[0]
                    if user_id == -1:
                        raise ValueError("User was not present in the training data.")

        ## calculating the latent factors
        # Theta = np.empty(self.k, dtype = ctypes.c_float)
        a_vec, counts_df = self._predict_factors(counts_df, random_seed, True)

        ## adding the data to the model
        if update_existing:
            self.A[user_id] = a_vec
        else:
            if self.reindex:
                new_id = self.user_mapping_.shape[0]
                self.user_mapping_ = np.r_[self.user_mapping_, user_id]
                if self.produce_dicts:
                    self.user_dict_[user_id] = new_id
            self.A = np.r_[self.A, a_vec.reshape((1, self.k))]
            self.nusers += 1

        ## updating the list of seen items for this user
        if self.keep_data:
            if update_existing:
                n_seen_by_user_before = self._n_seen_by_user[user_id]
                self._n_seen_by_user[user_id] = counts_df.shape[0]
                self._seen = np.r_[self._seen[:user_id], counts_df.ItemId.values, self._seen[(user_id + 1):]]
                self._st_ix_user[(user_id + 1):] += self._n_seen_by_user[user_id] - n_seen_by_user_before
            else:
                self._n_seen_by_user = np.r_[self._n_seen_by_user, counts_df.shape[0]]
                self._st_ix_user = np.r_[self._st_ix_user, self._seen.shape[0]]
                self._seen = np.r_[self._seen, counts_df.ItemId.values]

        return True
    
    def predict(self, user, item):
        """
        Predict count for combinations of users and items
        
        Note
        ----
        You can either pass an individual user and item, or arrays representing
        tuples (UserId, ItemId) with the combinatinons of users and items for which
        to predict (one row per prediction).

        Parameters
        ----------
        user : array-like (npred,) or obj
            User(s) for which to predict each item.
        item: array-like (npred,) or obj
            Item(s) for which to predict for each user.
        """
        assert self.is_fitted
        if isinstance(user, list) or isinstance(user, tuple):
            user = np.array(user)
        if isinstance(item, list) or isinstance(item, tuple):
            item = np.array(item)
        if user.__class__.__name__=='Series':
            user = user.values
        if item.__class__.__name__=='Series':
            item = item.values
            
        if isinstance(user, np.ndarray):
            if len(user.shape) > 1:
                user = user.reshape(-1)
            assert user.shape[0] > 0
            if self.reindex:
                if user.shape[0] > 1:
                    user = pd.Categorical(user, self.user_mapping_).codes
                else:
                    if self.user_dict_ is not None:
                        try:
                            user = self.user_dict_[user]
                        except:
                            user = -1
                    else:
                        user = pd.Categorical(user, self.user_mapping_).codes[0]
        else:
            if self.reindex:
                if self.user_dict_ is not None:
                    try:
                        user = self.user_dict_[user]
                    except:
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
                    if self.item_dict_ is not None:
                        try:
                            item = self.item_dict_[item]
                        except:
                            item = -1
                    else:
                        item = pd.Categorical(item, self.item_mapping_).codes[0]
        else:
            if self.reindex:
                if self.item_dict_ is not None:
                    try:
                        item = self.item_dict_[item]
                    except:
                        item = -1
                else:
                    item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
            item = np.array([item])

        assert user.shape[0] == item.shape[0]
        
        if user.shape[0] == 1:
            if (user[0] == -1) or (item[0] == -1):
                return np.nan
            else:
                return self.A[user].dot(self.B[item].T).reshape(-1)[0]
        else:
            nan_entries = (user == -1) | (item == -1)
            if np.any(nan_entries):
                if user.dtype != ctypes.c_size_t:
                    user = user.astype(ctypes.c_size_t)
                if item.dtype != ctypes.c_size_t:
                    item = item.astype(ctypes.c_size_t)
                out = np.empty(user.shape[0], dtype = ctypes.c_double)
                _predict_multiple(out, self.A, self.B, user, item, self.nthreads)
                return out
            else:
                non_na_user = user[~nan_entries]
                non_na_item = item[~nan_entries]
                out = np.empty(user.shape[0], dtype = ctypes.c_double)
                temp = np.empty(np.sum(~nan_entries), dtype = ctypes.c_double)
                _predict_multiple(temp, self.A, self.B, non_na_user.astype(ctypes.c_size_t), non_na_item.astype(ctypes.c_size_t), self.nthreads)
                out[~nan_entries] = temp
                out[nan_entries] = np.nan
                return out
        
    
    def topN(self, user, n=10, exclude_seen=True, items_pool=None):
        """
        Recommend Top-N items for a user

        Outputs the Top-N items according to score predicted by the model.
        Can exclude the items for the user that were associated to her in the
        training set, and can also recommend from only a subset of user-provided items.

        Note
        ----
        This function requires package 'hpfrec':
        https://www.github.com/david-cortes/hpfrec

        Parameters
        ----------
        user : obj
            User for which to recommend.
        n : int
            Number of top items to recommend.
        exclude_seen: bool
            Whether to exclude items that were associated to the user in the training set.
        items_pool: None or array
            Items to consider for recommending to the user.
        
        Returns
        -------
        rec : array (n,)
            Top-N recommended items.
        """
        try:
            from hpfrec import HPF
        except:
            self._throw_hpfrec_msg()
        temp = self
        temp.Theta = self.A
        temp.Beta = self.B
        temp.seen = self._seen
        return HPF.topN(temp, user, int(n), exclude_seen, items_pool)

    
    def eval_llk(self, input_df, full_llk=False):
        """
        Evaluate Poisson log-likelihood (plus constant) for a given dataset
        
        Note
        ----
        This Poisson log-likelihood is calculated only for the combinations of users and items
        provided here, so it's not a complete likelihood, and it might sometimes turn out to
        be a positive number because of this.
        Will filter out the input data by taking only combinations of users
        and items that were present in the training set.

        Note
        ----
        This function requires package 'hpfrec':
        https://www.github.com/david-cortes/hpfrec

        Parameters
        ----------
        input_df : pandas data frame (nobs, 3)
            Input data on which to calculate log-likelihood, consisting of IDs and counts.
            Must contain one row per non-zero observaion, with columns 'UserId', 'ItemId', 'Count'.
            If a numpy array is provided, will assume the first 3 columns
            contain that info.
        full_llk : bool
            Whether to calculate terms of the likelihood that depend on the data but not on the
            parameters. Ommitting them is faster, but it's more likely to result in positive values.

        Returns
        -------
        llk : dict
            Dictionary containing the calculated log-likelihood and the number of
            observations that were used to calculate it.
        """
        assert self.is_fitted
        try:
            from hpfrec import HPF, cython_loops
        except:
            self._throw_hpfrec_msg()
        HPF._process_valset(self, input_df, valset=False)
        out = {'llk': cython_loops.calc_llk(self.val_set.Count.values.astype(ctypes.c_float),
                                            self.val_set.UserId.values.astype(cython_loops.obj_ind_type),
                                            self.val_set.ItemId.values.astype(cython_loops.obj_ind_type),
                                            self.A.astype(ctypes.c_float),
                                            self.B.astype(ctypes.c_float),
                                            self.k,
                                            self.nthreads,
                                            bool(full_llk)),
               'nobs':self.val_set.shape[0]}
        del self.val_set
        return out


    def _throw_hpfrec_msg(self):
        install_msg  = "This function requires package 'hpfrec':\n"
        install_msg += "https://www.github.com/david-cortes/hpfrec\n"
        install_msg += "Can be installed with 'pip install hpfrec'."
        raise ValueError(install_msg)

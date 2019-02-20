extern "C" {
	#include <stddef.h>
	#include <R_ext/BLAS.h>
	void run_poismf(
		double *A, double *Xr, size_t *Xr_indptr, size_t *Xr_indices,
		double *B, double *Xc, size_t *Xc_indptr, size_t *Xc_indices,
		const size_t dimA, const size_t dimB, const size_t k,
		const double l2_reg, const double l1_reg, const int use_cg, double step_size,
		const size_t numiter, const size_t npass, const int ncores);
	void optimize_cg_single(
		double curr[], double X[], size_t X_ind[], size_t nnz_this,
		double F[], double Fsum[], int k, double l2_reg);
}
#include <Rcpp.h>
#ifdef _OPENMP
	#if _OPENMP < 20080101 /* OpenMP < 3.0 */
		#define size_t_for size_t
	#else
		#define size_t_for
	#endif
#else
	#define size_t_for size_t
#endif

// [[Rcpp::export]]
void r_wrapper_poismf(Rcpp::NumericVector A, Rcpp::NumericVector B, size_t dimA, size_t dimB, size_t k,
	Rcpp::NumericVector Xr, Rcpp::IntegerVector Xr_ind_int, Rcpp::IntegerVector Xr_indptr_int,
	Rcpp::NumericVector Xc, Rcpp::IntegerVector Xc_ind_int, Rcpp::IntegerVector Xc_indptr_int,
	size_t nnz, double l1_reg, double l2_reg, size_t niter, size_t npass, double step_size, int use_cg, int nthreads)
{
	/* Convert CSR and CSC matrix indices to size_t */
	std::vector<size_t> Xr_ind;
	std::vector<size_t> Xr_indptr;
	std::vector<size_t> Xc_ind;
	std::vector<size_t> Xc_indptr;
	Xr_indptr.reserve(dimA + 1);
	Xr_ind.reserve(nnz);
	Xc_indptr.reserve(dimB + 1);
	Xc_ind.reserve(nnz);

	#ifdef _OPENMP
		#if _OPENMP < 20080101 /* OpenMP < 3.0 */
			long i;
		#endif
	#endif

	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t_for i = 0; i < nnz; i++) { Xr_ind[i] = Xr_ind_int[i]; }
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t_for i = 0; i < nnz; i++) { Xc_ind[i] = Xc_ind_int[i]; }
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t_for i = 0; i < dimA + 1; i++) { Xr_indptr[i] = Xr_indptr_int[i]; }
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t_for i = 0; i < dimB + 1; i++) { Xc_indptr[i] = Xc_indptr_int[i]; }

	/* Run procedure */
	run_poismf(
		A.begin(), Xr.begin(), (size_t*) &Xr_indptr[0], (size_t*) &Xr_ind[0],
		B.begin(), Xc.begin(), (size_t*) &Xc_indptr[0], (size_t*) &Xc_ind[0],
		dimA, dimB, k,
		l2_reg, l1_reg, use_cg, step_size,
		niter, npass, nthreads);

	/* Note: C++ refuses to acknowledge that the vectors of type unsigned long are equivalent to size_t,
	   so don't use method .begin with the indices arrays */
}

// [[Rcpp::export]]
void predict_multiple(Rcpp::NumericVector A, Rcpp::NumericVector B, int k, size_t npred,
	Rcpp::IntegerVector ia, Rcpp::IntegerVector ib, Rcpp::NumericVector out, int nthreads)
{
	#ifdef _OPENMP
		#if _OPENMP < 20080101 /* OpenMP < 3.0 */
			long i;
		#endif
	#endif

	int one = 1;
	#pragma omp parallel for shared(npred, out, A, ia, B, ib, k) num_threads(nthreads)
	for (size_t_for i = 0; i < npred; i++) { out[i] = ddot_(&k, &A[ia[i] * k], &one, &B[ib[i] * k], &one); }
}


// [[Rcpp::export]]
void factorize_single(Rcpp::NumericVector a_vector, Rcpp::NumericVector x, Rcpp::IntegerVector ix, size_t nnz,
	Rcpp::NumericVector B, Rcpp::NumericVector Bsum, int k, double l2_reg)
{
	std::vector<size_t> ix_szt;
	ix_szt.reserve(nnz);

	for (size_t i = 0; i < nnz; i++) { ix_szt[i] = (size_t) ix[i] - 1; }

	optimize_cg_single(
		a_vector.begin(), x.begin(), (size_t*) &ix_szt[0], nnz,
		B.begin(), Bsum.begin(), k, l2_reg);
}

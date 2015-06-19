#include "bbvi.hpp"

void compute_mean_var(VecOfMat& list, arma::mat& mean, arma::mat& var) {
  auto n_rows = list[0]->n_rows;
  auto n_cols = list[0]->n_cols;
  mean.zeros(n_rows, n_cols);
  var.zeros(n_rows, n_cols);
  for(auto m : list) {
    mean += *m;
  }
  mean /= (list.size()+0.0);

  for(auto m : list) {
    var += (*m - mean) % (*m - mean);
  }
  var /= (list.size()+0.0);
}

shared_ptr<arma::mat> grad_bbvi_factorized(const pt::ptree& options,
                                           const VecOfMat& grad_log_q,
                                           const VecOfMat& log_p,
                                           const VecOfMat& log_q,
                                           BBVIStats& stats,
					   int threads) {
  auto n_rows = grad_log_q[0]->n_rows;
  auto n_cols = grad_log_q[0]->n_cols;
  size_t samples = options.get<int>("samples");
  size_t covariate_samples = max((size_t)10, samples / 4);
  assert(samples == grad_log_q.size());
  assert(samples > 10);
  
  // compute the gradient
  VecOfMat g_list;
  g_list.resize(samples);

#pragma omp parallel for num_threads(threads)
  for(arma::uword s=0; s<samples; ++s) {
    shared_ptr<arma::mat> g( new arma::mat() );
    if (log_q[s]) {
      *g = (*log_p[s] - *log_q[s]) % (*grad_log_q[s]);
    } else {
      *g = (*log_p[s]) % (*grad_log_q[s]);
    }
    g_list[s] = g;
  }

  // compute covariate
  arma::mat mean_g(n_rows, n_cols, arma::fill::zeros);
  arma::mat mean_glq(n_rows, n_cols, arma::fill::zeros);
  for(arma::uword s=0; s<covariate_samples; ++s) {
    mean_g += *g_list[s];
    mean_glq += *grad_log_q[s];
  }
  mean_g /= (covariate_samples+0.0);
  mean_glq /= (covariate_samples+0.0);

  arma::mat cov(n_rows, n_cols, arma::fill::zeros);
  arma::mat var_g(n_rows, n_cols, arma::fill::zeros);
  arma::mat var_glq(n_rows, n_cols, arma::fill::zeros);
  for(arma::uword s=0; s<covariate_samples; ++s) {
    // note that E[grad_log_q] = 0, use the improved estimates
    cov += (*g_list[s]) % (*grad_log_q[s]);
    var_g += (*g_list[s] - mean_g) % (*g_list[s] - mean_g);
    var_glq += (*grad_log_q[s]) % (*grad_log_q[s]);
  }
  cov /= (covariate_samples+0.0);
  var_g /= (covariate_samples+0.0);
  var_glq /= (covariate_samples+0.0);

  arma::mat rel_cov = cov % cov / var_g / var_glq;
  arma::mat a(n_rows, n_cols, arma::fill::zeros);
  for(arma::uword i=0; i<n_rows; ++i) {
    for(arma::uword j=0; j<n_cols; ++j) {
      auto rc = rel_cov(i,j);
      if (isfinite(rc) && (rc >= 0.5))
        a(i,j) = cov(i,j) / var_glq(i,j);
    }
  }

  // compute gradient
  VecOfMat g0_list, g1_list;
  g0_list.resize(samples - covariate_samples); g1_list.resize(samples - covariate_samples);
#pragma omp parallel for num_threads(threads)
  for(arma::uword s=covariate_samples; s<samples; ++s) {
    g0_list[s - covariate_samples] = g_list[s];
    shared_ptr<arma::mat> g1( new arma::mat(n_rows, n_cols) );
    *g1 = *g_list[s] - a % (*grad_log_q[s]);
    g1_list[s - covariate_samples] = g1;
  }
  arma::mat mean_g0, var_g0;
  compute_mean_var(g0_list, mean_g0, var_g0);    
  arma::mat mean_g1, var_g1;
  compute_mean_var(g1_list, mean_g1, var_g1);

  // statistics
  auto mean_all = [](const arma::mat& m) {
    return arma::mean(arma::mean(m));
  };
  stats.mean_sqr_g0 = mean_all(mean_g0 % mean_g0);
  stats.var_g0 = mean_all(var_g0);
  stats.mean_sqr_g1 = mean_all(mean_g1 % mean_g1);
  stats.var_g1 = mean_all(var_g1);

  return shared_ptr<arma::mat>( new arma::mat(mean_g1) );
}

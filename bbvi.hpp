#pragma once
#include "utils.hpp"

struct BBVIStats{
  // before control variates
  double mean_sqr_g0, var_g0;
  // after control variates
  double mean_sqr_g1, var_g1;

  BBVIStats()
    : mean_sqr_g0(), var_g0(), mean_sqr_g1(), var_g1(){
  }

  BBVIStats& operator+=(const BBVIStats& other) {
    mean_sqr_g0 += other.mean_sqr_g0;
    var_g0 += other.var_g0;
    mean_sqr_g1 += other.mean_sqr_g1;
    var_g1 += other.var_g1;
    return *this;
  }

  BBVIStats& operator/=(double x) {
    mean_sqr_g0 /= x;
    var_g0 /= x;
    mean_sqr_g1 /= x;
    var_g1 /= x;
    return *this;
  }

};

void compute_mean_var(VecOfMat& list, arma::mat& mean, arma::mat& var);

shared_ptr<arma::mat> grad_bbvi_factorized(const pt::ptree& options,
                                           const VecOfMat& grad_log_q,
                                           const VecOfMat& log_p,
                                           const VecOfMat& log_q,
                                           BBVIStats& stats,
					   int threads);

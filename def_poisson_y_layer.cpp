#include "def_poisson_y_layer.hpp"
#include "layer_factory.hpp"
#include <stdlib.h>


// example_ids are ONLY applied to y
DEFPoissonYLayer::LogPRowCol
DEFPoissonYLayer::log_p_row_column_sparse(shared_ptr<arma::mat> w,
                                   shared_ptr<arma::mat> z,
                                   const ExampleIds& example_ids) {
  assert(train_filter == NULL);
  auto n_examples = example_ids.size();
  shared_ptr<arma::rowvec> log_p_row( new arma::rowvec(n_examples, arma::fill::zeros) );
  shared_ptr<arma::vec> log_p_col( new arma::colvec(def_data->n_dim_y(), arma::fill::zeros) );
  shared_ptr<arma::sp_mat> y = def_data->get_sp_mat();

  for(size_t j=0; j<example_ids.size(); ++j) {
    arma::sp_vec ex = y->col(example_ids[j]);
    for(auto it=ex.begin(); it!=ex.end(); ++it) {
      auto i = it.row();
      double mu = lf->f(arma::dot(w->row(i).t(), z->col(j))) + poisson_rate_intercept;
      auto lp = (*it) * log(mu);
      (*log_p_row)(j) += lp;
      (*log_p_col)(i) += lp;
    }
  }

  *log_p_row -= arma::sum(*w, 0) * (*z);
  *log_p_col -= (*w) * arma::sum(*z, 1);
  LogPRowCol res;
  res.log_p_row_train = log_p_row;
  res.log_p_col_train = log_p_col;
  return res;
};

DEFPoissonYLayer::LogPRowCol
DEFPoissonYLayer::log_p_row_column_dense(bool add_log_factorial,
                                         shared_ptr<arma::mat> w,
                                         shared_ptr<arma::mat> z) {
  assert(train_filter != NULL);
  shared_ptr<arma::sp_mat> y = def_data->get_sp_mat();
  arma::mat mu = (*w) * (*z);
  arma::mat lpy = (*y) % arma::log(mu+poisson_rate_intercept) - mu;
  if (add_log_factorial) {
    lpy -= *log_factorial;
  }
  LogPRowCol res;
  res.log_p_row_train = shared_ptr<arma::rowvec>( new arma::rowvec(arma::sum(lpy % (*train_filter), 0)) );
  res.log_p_col_train = shared_ptr<arma::vec>( new arma::vec(arma::sum(lpy % (*train_filter), 1)) );
  res.log_p_row_test = shared_ptr<arma::rowvec>( new arma::rowvec(arma::sum(lpy % (1-*train_filter), 0)) );
  res.log_p_col_test = shared_ptr<arma::vec>( new arma::vec(arma::sum(lpy % (1-*train_filter), 1)) );
  return res;
}

shared_ptr<PredictionStats> DEFPoissonYLayer::prediction_stats(shared_ptr<arma::mat> w, shared_ptr<arma::mat> z) {
  PoissonPredictionStats* stats = new PoissonPredictionStats();
  assert(train_filter != NULL);
  shared_ptr<arma::sp_mat> y = def_data->get_sp_mat();
  arma::mat mu = (*w) * (*z); // words x users
  LOG(debug) << "row(y)=" << y->n_rows << " col(y)=" << y->n_cols
             << "\n" << "row(w)=" << w->n_rows << "col(w)=" << w->n_cols
             << "\n" << "roz(z)=" << z->n_rows << "col(z)=" << z->n_cols;
  arma::mat lpy = (*y) % arma::log(mu+poisson_rate_intercept) - mu;
  stats->train_ll = arma::accu((lpy - *log_factorial) % (*train_filter));
  stats->test_ll = arma::accu((lpy - *log_factorial) % (1 - *train_filter));
  stats->test_ll_no_log_fac = arma::accu((lpy) % (1 - *train_filter));
  
  // Normalize the mean
  
  // Log probability of the ones in the test set
  arma::mat log_mu = arma::log(mu);
  for (arma::uword c = 0; c < mu.n_cols; ++c) {
    log_mu.col(c) -= log(arma::sum(mu.col(c)));
  }
  arma::mat cond_log_prob((*y) % log_mu);

  stats->cond_test_ll_no_log_fac = arma::accu(cond_log_prob % (1 - *train_filter));
  double total_test_words = arma::accu((*y) % (1 - *train_filter));

  stats->cond_perplexity = exp(-stats->cond_test_ll_no_log_fac / total_test_words);
  return shared_ptr<PredictionStats>(stats);
}

void PoissonPredictionStats::pretty_print() const {
  printf("train_ll %.3e, test_ll %.3e, test_ll_no_log_fac %.3e, cond_perplexity %.3e, cond_test_ll_no_log_fac %.3e\n",
	 train_ll, test_ll, test_ll_no_log_fac, cond_perplexity, cond_test_ll_no_log_fac);
};

vector<string> DEFPoissonYLayer::prediction_header() const {
  vector<string> header = {"train_ll", "test_ll", "test_ll_no_log_fac", "cond_perplexity", 
			   "cond_test_ll_no_log_fac"};
  return header;
};

vector<string> PoissonPredictionStats::vals() const {
  char buffer[100];

  (void)buffer;
  auto d2s = [&](double d) {
    sprintf(buffer, "%.3e", d);
    return string(buffer);
  };

  vector<string> cols = { d2s(train_ll),
			  d2s(test_ll),
			  d2s(test_ll_no_log_fac),
			  d2s(cond_perplexity),
			  d2s(cond_test_ll_no_log_fac) };
  return cols;
};

vector<double> PoissonPredictionStats::numeric_vals() const {
  vector<double> vals = {train_ll, test_ll, test_ll_no_log_fac, cond_perplexity, cond_test_ll_no_log_fac};
  return vals;
};


REGISTER_P_Y_LAYER("poisson", DEFPoissonYLayer);

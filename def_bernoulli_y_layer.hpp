#pragma once
#include "utils.hpp"
#include "def_data.hpp"
#include "def_y_layer.hpp"
#include "def_layer.hpp"

class DEFBernoulliYLayer : public DEFYLayer {
private:
  pt::ptree options;
  // this is supposed to be a sparse matrix
  // but because train_filter is dense, not much to gain for efficiency
  shared_ptr<arma::mat> train_filter;
  shared_ptr<DEFData> def_data;
  ExampleIds all_examples;
public:

  DEFBernoulliYLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ), def_data( initializer.def_data ) {
    for(int j=0; j<def_data->n_examples(); ++j)
      all_examples.push_back(j);

    train_filter = def_data->get_train_filter();
  }

  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2,
                                      const ExampleIds& example_ids) {
    arma::mat w = (*z1) * (*z2);
    arma::mat y = def_data->get_mat()->cols(arma::uvec(example_ids));
    arma::mat lp(y.n_rows, y.n_cols, arma::fill::zeros);
    for(arma::uword j=0; j<y.n_cols; ++j) {
      for(arma::uword i=0; i<y.n_rows; ++i) {
        lp(i,j) = y(i,j) * (-softmax(-w(i,j))) + (1-y(i,j)) * (-softmax(w(i,j)));
      }
    }
    LogPRowCol res;
    res.log_p_row_train = shared_ptr<arma::rowvec>( new arma::rowvec(arma::sum(lp, 0)) );
    res.log_p_col_train = shared_ptr<arma::vec>( new arma::vec(arma::sum(lp, 1)) );
    return res;
  }

  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2) {
    return log_p_row_column(z1, z2, all_examples);
  }

  // we don't use train_filter for this application
  virtual LogPRowCol log_likelihood_row_column(shared_ptr<arma::mat> z1,
                                               shared_ptr<arma::mat> z2) {
    auto res = log_p_row_column(z1, z2);
    res.log_p_row_test = shared_ptr<arma::rowvec>( new arma::rowvec(z2->n_cols, arma::fill::zeros) );
    res.log_p_col_test = shared_ptr<arma::vec>( new arma::vec(z1->n_rows, arma::fill::zeros) );
    return res;
  }
};

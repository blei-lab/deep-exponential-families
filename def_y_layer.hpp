#pragma once

class PredictionStats {
public:
  virtual void pretty_print() const = 0;
  virtual vector<string> vals() const = 0;
  virtual vector<double> numeric_vals() const = 0;
};

class DEFYLayer {
public:

  struct LogPRowCol {
    shared_ptr<arma::rowvec> log_p_row_train;
    shared_ptr<arma::colvec> log_p_col_train;
    shared_ptr<arma::rowvec> log_p_row_test;
    shared_ptr<arma::colvec> log_p_col_test;  
  };



  // row/column sum of log p(y)
  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2,
                                      const ExampleIds& example_ids) = 0;

  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2) = 0;


  virtual LogPRowCol log_likelihood_row_column(shared_ptr<arma::mat> z1,
                                               shared_ptr<arma::mat> z2) = 0;


  virtual shared_ptr<PredictionStats> prediction_stats(shared_ptr<arma::mat> z1,
						       shared_ptr<arma::mat> z2) = 0;
 
  virtual vector<string> prediction_header() const = 0;
};

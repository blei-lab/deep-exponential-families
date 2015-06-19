#pragma once
#include "utils.hpp"
#include "link_function.hpp"
#include "def_data.hpp"
#include "def_y_layer.hpp"
#include "def_layer.hpp"

class DEFPoissonYLayer;
class PoissonPredictionStats : public PredictionStats {
public:
  virtual void pretty_print() const;
  virtual vector<string> vals() const;
  virtual vector<double> numeric_vals() const;
private:
  double train_ll, test_ll, test_ll_no_log_fac, cond_perplexity, cond_test_ll_no_log_fac;
  friend class DEFPoissonYLayer;
};

class DEFPoissonYLayer : public DEFYLayer {
private:
  pt::ptree options;
  double poisson_rate_intercept;
  // this is supposed to be a sparse matrix
  // but because train_filter is dense, not much to gain for efficiency
  shared_ptr<arma::mat> log_factorial;
  shared_ptr<arma::mat> train_filter;
  shared_ptr<DEFData> def_data;
  ExampleIds all_examples;
  LinkFunction* lf;
public:

  DEFPoissonYLayer(const pt::ptree& in_options, const DEFInitializer& initializer)
    : options( in_options ), def_data( initializer.def_data ) {
    poisson_rate_intercept = options.get<double>("y_layer.poisson_rate_intercept");
    // set default value
    if (options.get<string>("layer.lf", "") == "") {
      options.put<string>("layer.lf", "id");
    }
    // only support identify link for efficiency
    assert(options.get<string>("layer.lf") == "id");
    lf = get_link_function(options.get<string>("layer.lf"));
    for(int j=0; j<def_data->n_examples(); ++j)
      all_examples.push_back(j);

    train_filter = def_data->get_train_filter();

    // log factorial only deals with sparse data
    log_factorial = shared_ptr<arma::mat>( new arma::mat(*def_data->get_sp_mat()) );
    log_factorial->transform([](double v) {
        return v<=1 ? 0 : gsl_sf_lngamma(v+1); 
      });
    LOG(debug) << "log factorial row=" << log_factorial->n_rows
               << " col=" << log_factorial->n_cols;

    // output the sum of log_factorial for debugging purpose
    if (train_filter) {
      LOG(debug) << "train log factorial "
                 << arma::sum(arma::sum((*log_factorial) % (*train_filter)));
      LOG(debug) << "test log factorial "
                 << arma::sum(arma::sum((*log_factorial) % (1-*train_filter)));
    }
  }

  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2,
                                      const ExampleIds& example_ids) {
    if (train_filter == NULL) { // training, sparse data
      return log_p_row_column_sparse(z1, z2, example_ids);
    }
    else { // validation/test, with a filter
      assert(example_ids == all_examples);
      return log_p_row_column_dense(false, z1, z2);      
    }
  }

  virtual LogPRowCol log_p_row_column(shared_ptr<arma::mat> z1,
                                      shared_ptr<arma::mat> z2) {
    return log_p_row_column_dense(false, z1, z2);
  }

  virtual LogPRowCol log_likelihood_row_column(shared_ptr<arma::mat> z1,
                                               shared_ptr<arma::mat> z2) {
    return log_p_row_column_dense(true, z1, z2);
  }

  virtual shared_ptr<PredictionStats> prediction_stats(shared_ptr<arma::mat> z1,
						       shared_ptr<arma::mat> z2);

  virtual vector<string> prediction_header() const;

private:

  LogPRowCol log_p_row_column_sparse(shared_ptr<arma::mat> w,
                                     shared_ptr<arma::mat> z,
                                     const ExampleIds& example_ids);

  LogPRowCol log_p_row_column_dense(bool add_log_factorial,
                                    shared_ptr<arma::mat> w,
                                    shared_ptr<arma::mat> z);
};

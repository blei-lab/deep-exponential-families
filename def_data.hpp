#pragma once

#include "utils.hpp"
#include <gsl/gsl_sf_gamma.h>

// a general data base class
class DEFData {
public:
  // sp_mat || mat
  virtual string get_data_type() = 0;
  virtual shared_ptr<arma::sp_mat> get_sp_mat() {
    throw runtime_error("get_sp_mat() not implemented in DEFData");
    return NULL;
  }
  virtual shared_ptr<arma::mat> get_mat() {
    throw runtime_error("get_mat() not implemented in DEFData");
    return NULL;
  }

  // returns NULL by default, not NULL means there is a
  // training/testing split
  virtual shared_ptr<arma::mat> get_train_filter() {
    return NULL;
  }

  virtual int n_examples() = 0;
  virtual int n_dim_y() = 0;
  virtual shared_ptr<DEFData> transpose() const = 0;

  virtual void transform(std::function<double(double)> func) = 0;
};

// sparse-text | test-text
shared_ptr<DEFData> build_def_data(const string& data_type,
                                   const pt::ptree& options,
                                   const string& fname);

shared_ptr<arma::sp_mat> read_text_data(const string& fname, int max_examples);

class SparseTextData : public DEFData {
private:
  pt::ptree options;
  shared_ptr<arma::sp_mat> data;
  SparseTextData() {}

public:
  virtual string get_data_type() {
    return "sp_mat";
  }
  virtual shared_ptr<arma::sp_mat> get_sp_mat() {
    return data;
  }

  virtual int n_examples() {
    return data->n_cols;
  }

  virtual int n_dim_y() {
    return data->n_rows;
  }

  virtual shared_ptr<DEFData> transpose() const;

  SparseTextData(const pt::ptree& options, const string& fname)
    : options(options){
    auto max_examples = options.get<int>("max_examples");
    auto exp_fname = expand_environment_variables(fname);
    data = read_text_data(exp_fname, max_examples);
  }

  // NOTE: this function is slow
  shared_ptr<arma::sp_mat>
  slice_data(const ExampleIds& example_ids) {
    shared_ptr<arma::sp_mat> batch(new arma::sp_mat(data->n_rows,
                                                    example_ids.size()));
    for(size_t i=0; i<example_ids.size(); ++i)
      batch->col(i) = data->col(example_ids[i]);
    return batch;
  }

  void transform(std::function<double(double)> func) {
    for(auto it=data->begin(); it!=data->end(); ++it) {
      *it = func(*it);
    }
  }

};

class MaskedTextData : public DEFData {
private:
  pt::ptree options;

  shared_ptr<arma::sp_mat> data;
  shared_ptr<arma::mat> test_filter, train_filter;

  MaskedTextData() {}

public:  
  virtual string get_data_type() {
    return "sp_mat";
  }
  virtual shared_ptr<arma::sp_mat> get_sp_mat() {
    return data;
  }
  
  virtual int n_examples() {
    return data->n_cols;
  }

  virtual int n_dim_y() {
    return data->n_rows;
  }

  virtual shared_ptr<DEFData> transpose() const;

  virtual shared_ptr<arma::mat> get_train_filter() {
    return train_filter;
  }

  MaskedTextData(const pt::ptree& options, const string& fname);

  void transform(std::function<double(double)> func) {
    for(auto it=data->begin(); it!=data->end(); ++it) {
      *it = func(*it);
    }
  }

};

class DenseData : public DEFData {
private:
  pt::ptree options;
  shared_ptr<arma::mat> data;

  DenseData() {}

public:  
  virtual string get_data_type() {
    return "mat";
  }
  virtual shared_ptr<arma::mat> get_mat() {
    return data;
  }
  
  virtual int n_examples() {
    return data->n_cols;
  }

  virtual int n_dim_y() {
    return data->n_rows;
  }

  virtual shared_ptr<DEFData> transpose() const;

  DenseData(const pt::ptree& options, const string& fname);

  void transform(std::function<double(double)> func) {
    data->transform(func);
  }

};



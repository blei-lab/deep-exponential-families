#pragma once

#include <gsl/gsl_rng.h>
#include "utils.hpp"
#include "random.hpp"
#include "serialization.hpp"
#include "def.hpp"
#include "def_layer.hpp"
#include "def_y_layer.hpp"
#include "link_function.hpp"
#include "def_data.hpp"

class DEF2Model {

private:
  pt::ptree ptree;
  shared_ptr<DEFData> def_data;
  vector<GSLRandom*> vec_rng;

  int iteration;
  int layers;
  int batch_st;
  // train | test
  string model_type;
  int n_examples, n_dim_y, n_samples, n_dim_z_1;
  shared_ptr<ofstream> log_file;
  string data_file;

  // Assumes def_rows is per user, def_cols is per item 
  DEF def_rows, def_cols;
  shared_ptr<DEFYLayer> y_layer;
  ExampleIds full_col_ids;

  struct TrainStats {
    int iteration;
    shared_ptr<PredictionStats> prediction_stats;
    arma::vec lp_y, elbo;
    DEF::TrainStats row_stats, col_stats;

    TrainStats(int iteration, int row_layers, int col_layers, int samples) : 
      iteration(iteration), lp_y(samples), row_stats(iteration, row_layers, samples),
      col_stats(iteration, col_layers, samples), elbo(samples) {}
  };

  void print_stats(const TrainStats& stats);
  void log_stats(const TrainStats& stats, ofstream& of);
  void log_stats_header(ofstream& of, const vector<string>& prediction_header);
    
public:
  void set_full(bool full) const { def_rows.set_full(full); def_cols.set_full(full); }

  DEF2Model() : data_file("") {}
  DEF2Model(const string& data_file) : data_file(data_file) {}
  DEF2Model(const pt::ptree& ptree)
    : ptree(ptree), data_file("") {
    init();
  }
  void init();

  ~DEF2Model() {
    for (GSLRandom* r : vec_rng) {
      delete r;
    }
  }

  int get_iteration() const {
    return iteration;
  }

  void train_model();
  TrainStats train_batch(const ExampleIds& example_ids);
  TrainStats compute_log_likelihood();

  void copy_iteration(const DEF2Model& other) {
    iteration = other.iteration-1;
  }

  void copy_w_params(const DEF2Model& other) {
    def_rows.copy_w_params(other.def_rows);
    def_cols.copy_w_params(other.def_cols);
    def_cols.copy_z_params(other.def_cols); 
  }
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & ptree;

    ar & iteration;
    ar & batch_st;

    for(int i=0; i<n_samples; ++i) {
      ar & *vec_rng[i];
    }
    
    ar & def_rows;
    ar & def_cols;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & ptree;
    init();
    
    ar & iteration;
    ar & batch_st;

    for(int i=0; i<n_samples; ++i) {
      ar & *vec_rng[i];
    }

    ar & def_rows;
    ar & def_cols;
  }

  void load_part(shared_ptr<DEF2Model>part_model, int k) {
    assert(k >= 1);
    def_rows.load_part(part_model->def_rows, k);
    def_cols.load_part(part_model->def_cols, k);
  }
  void save_params(const string& fname) const;
  void load_params(const string& fname);
};

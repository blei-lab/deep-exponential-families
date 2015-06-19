#pragma once

#include <gsl/gsl_rng.h>
#include "utils.hpp"
#include "serialization.hpp"
#include "random.hpp"
#include "def.hpp"
#include "def_layer.hpp"
#include "def_y_layer.hpp"
#include "link_function.hpp"
#include "def_data.hpp"

class DEFModel {

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
  bool exp_fam_mode;

  DEF def;
  shared_ptr<DEFYLayer> y_layer;

  // for exponential distribution. q_w_obs - q_w_obs_b
  shared_ptr<DEFPriorLayer> w_obs_layer, w_obs_layer_b;
  shared_ptr<InferenceFactorizedLayer> q_w_obs_layer, q_w_obs_layer_b;


  struct TrainStats {
    int iteration;
    shared_ptr<PredictionStats> prediction_stats;
    arma::vec lp_y;

    BBVIStats w_obs_layer;
    arma::vec lq_w_obs;
    arma::vec lp_w_obs;

    DEF::TrainStats def_stats;
    TrainStats(int iteration, int layers, int samples) : iteration(iteration),
							 lp_y(samples), lq_w_obs(samples), lp_w_obs(samples),
							 def_stats(iteration, layers, samples) {}

    void fill_for_print(const string& model_type) const;
    mutable vector<const BBVIStats*> w_for_print;
    mutable vector<const arma::vec*> lp_w_for_print;
    mutable vector<const arma::vec*> lq_w_for_print;
  };

  void print_stats(const TrainStats& stats);
  void log_stats(const TrainStats& stats, ofstream& of);
  void log_stats_header(ofstream& of, const vector<string>& prediction_header);
    
public:
  void set_full(bool full) const { def.set_full(full); }

  DEFModel() : data_file("") {}
  DEFModel(const string& data_file) : data_file(data_file) {}
  DEFModel(const pt::ptree& ptree)
    : ptree(ptree), data_file("") {
    init();
  }
  void init();

  ~DEFModel() {
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

  void copy_iteration(const DEFModel& other) {
    iteration = other.iteration-1;
  }

  void copy_w_params(const DEFModel& other) {
    q_w_obs_layer->copy_params(&*other.q_w_obs_layer);
    if (other.q_w_obs_layer_b == NULL)
      q_w_obs_layer_b = NULL;
    else
      q_w_obs_layer_b->copy_params(&*other.q_w_obs_layer_b);
    def.copy_w_params(other.def);
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
    
    ar & def;
    InferenceFactorizedLayer* lp = q_w_obs_layer.get();
    ar & lp;
    if (q_w_obs_layer_b != NULL) {
      InferenceFactorizedLayer* lp_b = q_w_obs_layer_b.get();
      ar & lp_b;
    }
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

    ar & def;
    InferenceFactorizedLayer* lp;
    ar & lp;
    q_w_obs_layer.reset(lp);
    if (q_w_obs_layer_b != NULL) {
      InferenceFactorizedLayer* lp_b;
      ar & lp_b;
      q_w_obs_layer_b.reset(lp_b);
    }

  }

  void load_part(shared_ptr<DEFModel> part_model, int k) {
    assert(k >= 1);
    w_obs_layer = part_model->w_obs_layer;
    w_obs_layer_b = part_model->w_obs_layer_b;
    q_w_obs_layer = part_model->q_w_obs_layer;
    q_w_obs_layer_b = part_model->q_w_obs_layer_b;
    def.load_part(part_model->def, k);
  }

  void save_params(const string& fname) const;
  void load_params(const string& fname);
};

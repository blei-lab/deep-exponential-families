#pragma once

#include <gsl/gsl_rng.h>
#include "utils.hpp"
#include "def_layer.hpp"
#include "def_y_layer.hpp"
#include "link_function.hpp"
#include "def_data.hpp"
#include "serialization.hpp"

class DEF {

private:
  pt::ptree ptree;
  shared_ptr<DEFData> def_data;

  string name;  // The DEF's name in the property file

  int iteration;
  int layers;
  int batch_st;
  bool exp_fam_mode;
  mutable bool full;

  // train | test
  string model_type;
  vector<int> layer_sizes;
  vector<string> qz_types;
  int n_examples, n_dim_y, n_samples;
  shared_ptr<ofstream> log_file;

  // whether each z has a bias term
  vector<bool> has_bias;
  shared_ptr<DEFPriorLayer> prior_z_layer;
  vector<shared_ptr<DEFPriorLayer> > prior_z_bias_layers;
  vector<shared_ptr<DEFLayer> > pz_layers;
  vector<shared_ptr<DEFPriorLayer> > prior_w_layers, prior_w_layers_b;
  vector<shared_ptr<InferenceFactorizedLayer> > qw_layers, qw_layers_b;
  vector<shared_ptr<InferenceFactorizedLayer> > qz_layers, qz_bias_layers;
  void reset();

  struct SampleState {
    vector<VecOfMat> z_samples, z_bias_samples;
    // w_ab = w_a - w_b
    vector<VecOfMat> w_samples, w_samples_b, w_samples_ab;
    vector<VecOfMat> samples_log_pz, samples_log_pz_bias;
    vector<VecOfMat> samples_log_qz, samples_log_qz_bias;
    vector<VecOfMat> samples_log_pw, samples_log_pw_b;
    vector<VecOfMat> samples_log_qw, samples_log_qw_b;
    vector<VecOfCube> samples_score_qz, samples_score_qz_bias;
    vector<VecOfCube> samples_score_qw, samples_score_qw_b;

    // has_w_b: has b layer for the weights
    void init(int samples, int n_layers, int n_w_layers, bool exp_family_mode, bool has_w_b) {
      auto expand_vec_of_mat = [=](vector<VecOfMat>& v, int num_layers) {
	v.resize(num_layers);
	for(auto& l : v)
	  l.resize(samples);
      };

      auto expand_vec_of_cube = [=](vector<VecOfCube>& v, int num_layers) {
	v.resize(num_layers);
	for(auto& l : v)
	  l.resize(samples);
      };

      expand_vec_of_mat(z_samples, n_layers);
      expand_vec_of_mat(samples_log_pz, n_layers);
      expand_vec_of_cube(samples_score_qz, n_layers);

      // z_bias layers
      expand_vec_of_mat(z_bias_samples, n_layers);
      expand_vec_of_mat(samples_log_pz_bias, n_layers);
      expand_vec_of_mat(samples_log_qz_bias, n_layers);
      expand_vec_of_cube(samples_score_qz_bias, n_layers);

      expand_vec_of_mat(w_samples, n_w_layers);
      expand_vec_of_mat(samples_log_pw, n_w_layers);
      expand_vec_of_cube(samples_score_qw, n_w_layers);

      expand_vec_of_mat(samples_log_qz, n_layers);
      expand_vec_of_mat(samples_log_qw, n_w_layers);

      if (has_w_b) {
        expand_vec_of_mat(w_samples_b, n_w_layers);
        expand_vec_of_mat(w_samples_ab, n_w_layers);
        expand_vec_of_mat(samples_log_pw_b, n_w_layers);
        expand_vec_of_cube(samples_score_qw_b, n_w_layers);
        expand_vec_of_mat(samples_log_qw_b, n_w_layers);
      }
    }

    SampleState() {}
  } sample_state;

  void init();

public:
  void set_full(bool full) const { this->full = full; }

  struct TrainStats {
    int iteration;
    arma::vec elbo;

    // layer x sample x 1
    vector<arma::vec> lp_z;
    vector<arma::vec> lq_z;
    vector<BBVIStats> bbvi_stats_z;

    // TODO lp_w_b, lq_w_b
    vector<arma::vec> lp_w;
    vector<arma::vec> lq_w;
    vector<BBVIStats> bbvi_stats_w;

    TrainStats(int iteration, int layers, int samples)
      : iteration( iteration ),
	elbo(samples, arma::fill::zeros),
	lp_z(layers), lq_z(layers), lp_w(layers - 1), lq_w(layers - 1) {
      for(auto& v : lp_z)
	v.set_size(samples);
      for(auto& v : lp_w)
	v.set_size(samples);
      for(auto& v : lq_z)
	v.set_size(samples);
      for(auto& v : lq_w)
	v.set_size(samples);
    }
  };

  size_t num_layers() const { return layer_sizes.size(); }
  const vector<int>& get_layer_sizes() const { return layer_sizes; }
  int get_iteration() const { return iteration; }

  void init_with_params(const pt::ptree& ptree, const string& name, shared_ptr<DEFData> data);

  // Initializes the state for sampling
  void prepare_to_sample(const ExampleIds& example_ids);

  // Saves a bunch of internal state for the samples taken and returns lowest layer sample
  shared_ptr<arma::mat> sample(const ExampleIds& example_ids, int sample_index, gsl_rng* rng, TrainStats* stats);

  // Update parameters given p_y. There is one entry per training example for each sample in p_y
  void update(const ExampleIds& example_ids, const vector<shared_ptr<arma::rowvec> >& p_y, DEF::TrainStats* stats);

  // Returns the mean of the lowest layer of z's
  shared_ptr<arma::mat> mean() const;

  void copy_w_params(const DEF& other) {
    // copy the iteration number as well
    iteration = other.iteration-1;
    for(size_t l=0; l< layer_sizes.size() -1; ++l) {
      qw_layers[l]->copy_params(&*other.qw_layers[l]);
      // the b layers
      if (qw_layers_b.size() > l) {
        qw_layers_b[l]->copy_params(&*other.qw_layers_b[l]);
      }

      // the bias for the z
      if (qz_bias_layers[l] != NULL) {
        qz_bias_layers[l]->copy_params(&*other.qz_bias_layers[l]);
      }

    }
  }

  void copy_z_params(const DEF& other) {
    // copy the iteration number as well
    iteration = other.iteration-1;
    for(size_t l=0; l< layer_sizes.size(); ++l) {
      qz_layers[l]->copy_params(&*other.qz_layers[l]);
    }
  }

  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & ptree;
    ar & name;

    ar & iteration;
    ar & batch_st;

    ar & full;

    if (n_examples <= ptree.get<int>("serialization_max_examples") || full) {
      for (auto l : qz_layers) {
	InferenceFactorizedLayer* lp = l.get();
	ar & lp;
      }
    }
    for (auto l : qw_layers) {
      InferenceFactorizedLayer* lp = l.get();
      ar & lp;
    }

    if (ptree.get_child(name+"_weights").get<string>("type") == "exp")
      assert(qw_layers_b.size() == qw_layers.size());
    else
      assert(qw_layers_b.size() == 0);

    for (auto l : qw_layers_b) {
      InferenceFactorizedLayer* lp = l.get();
      ar & lp;
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & ptree;
    ar & name;

    init();

    ar & iteration;
    ar & batch_st;

    ar & full;

    // This is inefficient. The layers are inited correctly earlier, but we overwrite them here.
    if (n_examples <= ptree.get<int>("serialization_max_examples") || full) {
      for (auto& l : qz_layers) {
	InferenceFactorizedLayer* lp;
        ar & lp;
	l.reset(lp);
      }
    }
    for (auto& l : qw_layers) {
      InferenceFactorizedLayer* lp;
      ar & lp;
      l.reset(lp);
    }

    if (ptree.get_child(name+"_weights").get<string>("type") == "exp") {
      assert(qw_layers_b.size() == qw_layers.size());
      for (auto& l : qw_layers_b) {
        InferenceFactorizedLayer* lp;
        ar & lp;
        l.reset(lp);
      }
    }
  }

  // For Python?
  void save_params(FILE* ofile) const;

  // DEPRECATED!
  void load_params(FILE* ifile);

  // load the first k layers of z and the first k-1 layers of w from part_model
  void load_part(DEF& part_model, int k) {

    if (n_examples <= ptree.get<int>("serialization_max_examples") || full) {
      for(int i=0; i<k && i<qz_layers.size(); ++i) {
        qz_layers[i] = part_model.qz_layers[i];
      }
    }

    for(int i=0; i<k-1; ++i) {
      qw_layers[i] = part_model.qw_layers[i];
    }

    if (ptree.get_child(name+"_weights").get<string>("type") == "exp") {
      assert(qw_layers_b.size() == qw_layers.size());
      for(int i=0; i<k-1; ++i) {
        qw_layers_b[i] = part_model.qw_layers_b[i];
      }
    }
  }

};

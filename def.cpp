#include <boost/optional/optional.hpp>
#include "def_model.hpp"
#include "layer_factory.hpp"

#include <boost/archive/impl/text_iarchive_impl.ipp>

using boost::optional;

void DEF::init_with_params(const pt::ptree& ptree, const string& name, shared_ptr<DEFData> data) {
  this->ptree = ptree;
  this->name = name;
  this->def_data = data;
  init();
}

void DEF::reset() {
  layer_sizes.clear();
  qz_types.clear();
  pz_layers.clear();
  prior_w_layers.clear();
  qw_layers.clear();
  prior_w_layers_b.clear();
  qw_layers_b.clear();
  qz_layers.clear();
}

void DEF::init() {
  reset();

  assert(def_data);
  n_samples = ptree.get<int>("samples");
  exp_fam_mode = ptree.get<bool>("exp_fam_mode");

  iteration = 0;
  layers = 0;
  batch_st = 0;

  // data
  model_type = ptree.get<string>("model_type");
  auto data_type = ptree.get<string>(model_type + ".data_type");
  n_examples = def_data->n_examples();
  n_dim_y = def_data->n_dim_y();
    
  LOG(debug) << "n_examples " << n_examples;
  LOG(debug) << "serialization_max_examples " << ptree.get<int>("serialization_max_examples");

  // We use a dummy rng 
  shared_ptr<GSLRandom> init_rng(new GSLRandom());
  auto seed = ptree.get<int>("seed");
  gsl_rng_set(init_rng->rng, seed);  

  DEFInitializer initializer;
  initializer.rng = init_rng->rng;
  initializer.def_data = def_data;

  string name(this->name);

  // prior(z) | prior(w) layer
  auto build_prior_layer = [&](const string& layer_name) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(layer_name));
    auto layer_type = options.get<string>("layer.type");
    return GET_PRIOR_LAYER(layer_type, options, initializer);
  };

  // prior_layer is parameterized symmetrically
  // so prior_bias is the same with prior_z
  auto build_prior_z_bias_layer = build_prior_layer;

  // p(z) layer
  auto build_pz_layer = [&](const string& layer_name) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(layer_name));
    auto layer_type = options.get<string>("layer.type");
    return GET_P_Z_LAYER(layer_type, options, initializer);
  };

  // q(z) layer
  auto build_qz_factorized_layer = [&](const string& layer_name) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(layer_name));
    options.put("n_examples", n_examples);
    // used for initialization
    options.put("lf", options.get<string>("layer.q_lf"));
    auto layer_type = options.get<string>("layer.type");
    return GET_Q_LAYER(layer_type, options, initializer);
  };

  // qz_bias layer, only one example
  auto build_qz_bias_factorized_layer = [&](const string& layer_name) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(layer_name));
    // only 1 example
    options.put("n_examples", 1);
    auto layer_type = options.get<string>("layer.type");
    return GET_Q_LAYER(layer_type, options, initializer);
  };

  // q(w) layer
  auto build_qw_layer = [&](int z_lower, int z_higher) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(name + "_weights"));
    options.put("n_examples", z_higher);
    options.put("layer.size", z_lower);
    //options.put("rng", init_rng);
    options.put("lf", options.get<string>("layer.q_lf"));
    auto layer_type = options.get<string>("layer.type");
    return GET_Q_LAYER(layer_type, options, initializer);
  };

  auto add_qz = [&](const string& layer_name, const string& q_type,
                    int z_lower, int z_higher) {
    if (q_type == "factorized") {
      qz_layers.push_back( build_qz_factorized_layer(layer_name) );
    } else {
      throw runtime_error("unknown q_type");
    }
  };

  // get qz type for a layer, can be overridden for specific model type
  auto get_qz_type = [&](int l, const string& layer_name) {
    auto q_type = ptree.get<string>(layer_name + ".q_type");
    LOG(debug) << "testing " << model_type + ".layer"+to_string(l) + "_q_type";
    if (ptree.get_optional<string>(model_type + ".layer"+to_string(l) + "_q_type")) {
      q_type = ptree.get<string>(model_type + ".layer"+to_string(l) + "_q_type");
      LOG(debug) << "change layer " << l << " type to " << q_type;
    }
    return q_type;
  };

  // z1 .. z_{L-1}
  int z_lower = n_dim_y;
  int z_higher = -1;
  bool make_w_layers = false;
  for (int i=1; i<4; ++i, ++layers) {
    auto layer_name = name + "_layer" + to_string(i);
    optional<pt::ptree&> child = ptree.get_child_optional(layer_name);
    if (!child)
      break;

    z_higher = ptree.get<int>(layer_name + ".size");
    layer_sizes.push_back(z_higher);

    auto q_type = get_qz_type(qz_types.size(), layer_name);
    if ((qz_types.size() == 0) &&
        ((model_type == "valid") || (model_type == "test"))) {
      q_type = "factorized";
    }
    qz_types.push_back(q_type);

    pz_layers.push_back( build_pz_layer(layer_name) );
    add_qz(layer_name, q_type, z_lower, z_higher);
    if (make_w_layers) {
      prior_w_layers.push_back( build_prior_layer(name + "_weights") );
      qw_layers.push_back( build_qw_layer(z_lower, z_higher) );
      if (ptree.get_child(name + "_weights").get<string>("type") == "exp") {
        prior_w_layers_b.push_back( build_prior_layer(name + "_weights") );
        qw_layers_b.push_back( build_qw_layer(z_lower, z_higher) );
      }
    }
    z_lower = z_higher;
    make_w_layers = true;

    // the bias term on z
    if (ptree.get_child_optional(layer_name + "_bias")) {
      prior_z_bias_layers.push_back( build_prior_z_bias_layer(layer_name + "_bias") );
      qz_bias_layers.push_back( build_qz_bias_factorized_layer(layer_name + "_bias") );
      LOG(debug) << "z_bias for layer " << i;
    }
    else {
      prior_z_bias_layers.push_back( NULL );
      qz_bias_layers.push_back( NULL );
    }
  }

  // z_L
  {
    string layer_name = name + "_prior_layer";
    z_higher = ptree.get<int>(layer_name + ".size");
    layer_sizes.push_back(z_higher);

    auto q_type = get_qz_type(qz_types.size(), layer_name);
    if ((qz_types.size() == 0) &&
        ((model_type == "valid") || (model_type == "test"))) {
      q_type = "factorized";
    }
    qz_types.push_back(q_type);

    prior_z_layer = build_prior_layer(layer_name);
    add_qz(layer_name, q_type, z_lower, z_higher);
    if (make_w_layers) {
      prior_w_layers.push_back( build_prior_layer(name + "_weights") );
      qw_layers.push_back( build_qw_layer(z_lower, z_higher) );
      if (ptree.get_child(name + "_weights").get<string>("type") == "exp") {
        prior_w_layers_b.push_back( build_prior_layer(name + "_weights") );
        qw_layers_b.push_back( build_qw_layer(z_lower, z_higher) );
      }
    }
    ++layers;
  }

  {
    printf("model sizes:");
    for(auto s : layer_sizes) {
      printf("%d ", s);
    }
    printf("\n");
  }
  // TODO only set `has_qw_b` when DEF_weights is `exp` (exponential distribution)
  bool has_qw_b = ptree.get_child(name+"_weights").get<string>("type") == "exp";
  sample_state.init(n_samples, layers, layers - 1, exp_fam_mode, has_qw_b);
}

void DEF::prepare_to_sample(const ExampleIds& example_ids) {
  size_t n_layers = layers;
  for(size_t l=0; l<n_layers; ++l) {
    if (qz_types[l] == "factorized")
      qz_layers[l]->truncate(example_ids);
    if (LOG_IS_ON(trace)) {
      LOG(debug) << "check q_z[" << l << "]";
      qz_layers[l]->check_params();
    }
  }
  for (size_t l = 0; l < n_layers - 1; ++l) {
    qw_layers[l]->truncate();
    if (LOG_IS_ON(trace)) {
      LOG(debug) << "check q_w[" << l << "]";
      qw_layers[l]->check_params();
    }

    // truncate w_b layers
    if (qw_layers_b.size() > l) {
      qw_layers_b[l]->truncate();
      if (LOG_IS_ON(trace)) {
        LOG(debug) << "check q_w_b[" << l << "]";
        qw_layers_b[l]->check_params();
      }
    }

    // truncate the only example for bias term
    if (qz_bias_layers[l] != NULL) {
      qz_bias_layers[l]->truncate();
      if (LOG_IS_ON(trace)) {
        LOG(debug) << "check q_z_bias[" << l << "]";
        qz_bias_layers[l]->check_params();
      }
    }

  }
  full = false;
}

shared_ptr<arma::mat> DEF::sample(const ExampleIds& example_ids, int sample_index, gsl_rng* rng, DEF::TrainStats* stats) {
  int s = sample_index;
  // sample w & log p(w) and log q(w) & grad log q(w)
  LOG(trace) << "sample w & log p(w) & log q(w) & grad log q(w)";
  size_t n_layers = (size_t) layers;
  assert(layers - 1 >= 0);

  for(size_t l=0; l< n_layers -1; ++l) {
    sample_state.w_samples[l][s] = qw_layers[l]->sample_matrix(rng);
    sample_state.samples_score_qw[l][s] = qw_layers[l]->grad_lq_matrix(sample_state.w_samples[l][s]);

    sample_state.samples_log_pw[l][s] = prior_w_layers[l]->log_p_matrix(sample_state.w_samples[l][s]);
    sample_state.samples_log_qw[l][s] = qw_layers[l]->log_q_matrix(sample_state.w_samples[l][s]);

    // the w_b layers
    if (prior_w_layers_b.size() > l) {
      sample_state.w_samples_b[l][s] = qw_layers_b[l]->sample_matrix(rng);
      sample_state.samples_score_qw_b[l][s] = qw_layers_b[l]->grad_lq_matrix(sample_state.w_samples_b[l][s]);

      sample_state.samples_log_pw_b[l][s] = prior_w_layers_b[l]->log_p_matrix(sample_state.w_samples_b[l][s]);
      sample_state.samples_log_qw_b[l][s] = qw_layers_b[l]->log_q_matrix(sample_state.w_samples_b[l][s]);
    }
  }

  // sample z & log q(z) & grad log q(z)
  for(size_t l=0; l<n_layers; ++l) {
    if (qz_types[l] == "factorized") {
      sample_state.z_samples[l][s] = qz_layers[l]->sample_matrix(rng, example_ids);
      auto score_qz = qz_layers[l]->grad_lq_matrix(sample_state.z_samples[l][s], example_ids);
      sample_state.samples_score_qz[l][s] = score_qz;
      auto lq_z = qz_layers[l]->log_q_matrix(sample_state.z_samples[l][s], example_ids);
      sample_state.samples_log_qz[l][s] = lq_z;
    }
  }

  double sampling_ratio = (example_ids.size()+0.0) / n_examples;
  
  // renormalize log p(w) & log q(w)
  for(size_t l=0; l<n_layers -1; ++l) {
    *(sample_state.samples_log_pw[l][s]) *= sampling_ratio;
    *(sample_state.samples_log_qw[l][s]) *= sampling_ratio;

    // renormalize w_b samples
    if (sample_state.samples_log_pw_b.size() > l) {
      *(sample_state.samples_log_pw_b[l][s]) *= sampling_ratio;
      *(sample_state.samples_log_qw_b[l][s]) *= sampling_ratio;
    }

    // the z_bias layers
    if (qz_bias_layers[l] != NULL) {
      sample_state.z_bias_samples[l][s] = qz_bias_layers[l]->sample_matrix(rng);
      sample_state.samples_log_pz_bias[l][s] = prior_z_bias_layers[l]->log_p_matrix(sample_state.z_bias_samples[l][s]);
      sample_state.samples_log_qz_bias[l][s] = qz_bias_layers[l]->log_q_matrix(sample_state.z_bias_samples[l][s]);

      auto score_qz_bias = qz_bias_layers[l]->grad_lq_matrix(sample_state.z_bias_samples[l][s]);
      sample_state.samples_score_qz_bias[l][s] = score_qz_bias;

      *(sample_state.samples_log_pz_bias[l][s]) *= sampling_ratio;
      *(sample_state.samples_log_qz_bias[l][s]) *= sampling_ratio;
    }
  }
  
  // compute log p(z|z_higher)
  for(size_t l=0; l<n_layers-1; ++l) {
    auto z = sample_state.z_samples[l][s];
    auto w_a = sample_state.w_samples[l][s];
    auto w = sample_state.w_samples[l][s];
    // w = w_a - w_b
    if (sample_state.w_samples_b.size() > l) {
      auto w_b = sample_state.w_samples_b[l][s];
      sample_state.w_samples_ab[l][s] = arma_sub(*sample_state.w_samples[l][s],
                                                 *sample_state.w_samples_b[l][s]);
      w = sample_state.w_samples_ab[l][s];
    }

    auto z_higher = sample_state.z_samples[l+1][s];
    shared_ptr<arma::mat> z_bias = NULL;
    if (prior_z_bias_layers[l] != NULL) {
      z_bias = sample_state.z_bias_samples[l][s];
    }
    auto lp_z = pz_layers[l]->log_p_matrix(w, z, z_higher, z_bias);
    sample_state.samples_log_pz[l][s] = lp_z;
  }
  // prior z layer
  {
    auto lp_z = prior_z_layer->log_p_matrix(sample_state.z_samples[n_layers-1][s]);
    sample_state.samples_log_pz[n_layers-1][s] = lp_z;
  }
  
  for(size_t l=0; l<n_layers; ++l) {
    stats->elbo(s) += (stats->lp_z[l](s) = arma::accu(*sample_state.samples_log_pz[l][s]));
    
    stats->elbo(s) -= (stats->lq_z[l](s) = arma::accu(*sample_state.samples_log_qz[l][s]));
    if (l < n_layers - 1) {
      stats->elbo(s) += (stats->lp_w[l](s) = arma::accu(*sample_state.samples_log_pw[l][s]));
      stats->elbo(s) -= (stats->lq_w[l](s) = arma::accu(*sample_state.samples_log_qw[l][s]));
      if (sample_state.samples_log_pw_b.size() > l) {
        stats->elbo(s) += (stats->lp_w[l](s) = arma::accu(*sample_state.samples_log_pw_b[l][s]));
        stats->elbo(s) -= (stats->lq_w[l](s) = arma::accu(*sample_state.samples_log_qw_b[l][s]));
      }
    }
  }
  return sample_state.z_samples[0][s]; // Return the lowest layer
}

void DEF::update(const ExampleIds& example_ids, const vector<shared_ptr<arma::rowvec> >& p_y, DEF::TrainStats* stats) {
  int samples = n_samples;
  const vector<shared_ptr<arma::rowvec> >& samples_lpy_row = p_y;
  auto threads = ptree.get<int>("threads");
  // BBVI for w
  if (model_type == "train") {
    stats->bbvi_stats_w.resize(qw_layers.size());
    for (size_t l=0; l<qw_layers.size(); ++l) {
      auto z_lower = layer_sizes[l];
      auto z_higher = layer_sizes[l+1];
      VecOfMat lp_list; lp_list.resize(samples);
      #pragma omp parallel for num_threads(threads)
      for(int s=0; s<samples; ++s) {
        shared_ptr<arma::mat> lp( new arma::mat(z_lower, z_higher) );
        lp->each_col() = arma::sum(*sample_state.samples_log_pz[l][s], 1);
	*lp += *sample_state.samples_log_pw[l][s];
	lp_list[s] = lp;
      }

      BBVIStats w_stats_l;
      w_stats_l = qw_layers[l]->update(sample_state.samples_score_qw[l], lp_list,
				       sample_state.samples_log_qw[l]);
      stats->bbvi_stats_w[l] = w_stats_l;

      // update z_bias, this is like a global parameter term
      if (qz_bias_layers[l] != NULL) {
        VecOfMat lp_zbias_list; lp_zbias_list.resize(samples);
#pragma omp parallel for num_threads(threads)
        for(int s=0; s<samples; ++s) {
  	  shared_ptr<arma::mat> lp_zbias( new arma::mat(z_lower, 1) );
  	  lp_zbias->each_col() = arma::sum(*sample_state.samples_log_pz[l][s], 1);
          *lp_zbias += *sample_state.samples_log_pz_bias[l][s];
          lp_zbias_list[s] = lp_zbias;
        }

        qz_bias_layers[l]->update(sample_state.samples_score_qz_bias[l], lp_zbias_list,
                                  sample_state.samples_log_qz_bias[l]);
      }

      if (qw_layers_b.size() > l) {
      // update lp
#pragma omp parallel for num_threads(threads)
        for(int s=0; s<samples; ++s) {
          *lp_list[s] -= *sample_state.samples_log_pw[l][s];
          *lp_list[s] += *sample_state.samples_log_pw_b[l][s];
        }
        BBVIStats w_stats_l_b = qw_layers_b[l]->update(sample_state.samples_score_qw_b[l], lp_list,
                                                       sample_state.samples_log_qw_b[l]);
        // stats->bbvi_stats_w_b[l] = w_stats_l_b;
      }

    }
  }

  size_t n_layers = qz_layers.size();
  size_t n_examples = example_ids.size();
  stats->bbvi_stats_z.resize(qz_layers.size());

  // BBVI for z
  for (size_t l=0; l<qz_layers.size(); ++l) {
    BBVIStats z_stats_l;
    if (qz_types[l] == "factorized") {
      // sum up log p term for bbvi
      VecOfMat lp_list;
      lp_list.resize(samples);

      #pragma omp parallel for num_threads(threads)
      for(int s=0; s<samples; ++s) {
        shared_ptr<arma::mat> lp( new arma::mat(layer_sizes[l],
                                                samples_lpy_row[s]->n_cols) );
	if (l == 0) {
	  lp->each_row() = *samples_lpy_row[s];
	} else {
	  lp->each_row() = arma::sum(*sample_state.samples_log_pz[l-1][s], 0);
	}
        *lp += *sample_state.samples_log_pz[l][s];
        lp_list[s] = lp;
      }
      z_stats_l = qz_layers[l]->update(sample_state.samples_score_qz[l], lp_list,
				       sample_state.samples_log_qz[l], example_ids);
    }
    stats->bbvi_stats_z[l] = z_stats_l;
  } // bbvi for L layers
}

shared_ptr<arma::mat> DEF::mean() const {
  shared_ptr<arma::mat> z_mean;
  z_mean = qz_layers[0]->mean_matrix();
  return z_mean;
}

void DEF::save_params(FILE* ofile) const {
  LOG(debug) << "save: serialization_max_examples " << ptree.get<int>("serialization_max_examples");
  auto max_examples = ptree.get<int>("serialization_max_examples");
  for(size_t l=0; l<layer_sizes.size(); ++l) {
    qz_layers[l]->save_params(ofile, max_examples);
  }
  for (auto l : qw_layers) {
    l->save_params(ofile);
  }
  for (auto l : qw_layers_b) {
    l->save_params(ofile);
  }
}

void DEF::load_params(FILE* ifile) {
  auto max_examples = ptree.get<int>("serialization_max_examples");
  for(size_t l=0; l<layer_sizes.size(); ++l) {
    qz_layers[l]->load_params(ifile, max_examples);
  }
  for (auto l : qw_layers) {
    l->load_params(ifile);
  }
  for (auto l : qw_layers_b) {
    l->load_params(ifile);
  }
}

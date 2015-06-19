#include <boost/optional/optional.hpp>
#include "def_model.hpp"
#include "layer_factory.hpp"

using boost::optional;

void DEFModel::TrainStats::fill_for_print(const string& model_type) const {
  if (model_type == "train") {
    w_for_print.push_back(&w_obs_layer);
  }
  lp_w_for_print.push_back(&lp_w_obs);
  lq_w_for_print.push_back(&lq_w_obs);

  for (size_t l=0; l < def_stats.bbvi_stats_w.size(); ++l) {
    w_for_print.push_back(&def_stats.bbvi_stats_w[l]);
  }

  for (size_t l=0; l < def_stats.lp_w.size(); ++l) {
    lp_w_for_print.push_back(&def_stats.lp_w[l]);
    lq_w_for_print.push_back(&def_stats.lq_w[l]);
  }
}

void DEFModel::print_stats(const TrainStats& stats) {
  stats.fill_for_print(model_type);
  printf("Iteration %d, ELBO %.3e, std %.3e, lp(y) %.3e\n",
         stats.iteration, arma::mean(stats.def_stats.elbo), arma::stddev(stats.def_stats.elbo),
         arma::mean(stats.lp_y));
  if (model_type != "train") {
    printf("\t%s_metric: ", model_type.c_str());
    stats.prediction_stats->pretty_print();
  }

  for(size_t l=0; l<def.get_layer_sizes().size(); ++l) {
    printf("\tLayer %d, lp_z %.3e, lp_w %.3e, lq_z %.3e, lq_w %.3e\n", (int)l,
           arma::mean(stats.def_stats.lp_z[l]), arma::mean(*stats.lp_w_for_print[l]),
           arma::mean(stats.def_stats.lq_z[l]), arma::mean(*stats.lq_w_for_print[l]));
    if (l < stats.w_for_print.size()) {
      auto & ws = *stats.w_for_print[l];
      printf("\tmean_sqr(g_w0) %.3e, mean_var(g_w0) %.3e, mean_sqr(g_w1) %.3e, mean_var(g_w1) %.3e\n",
             ws.mean_sqr_g0, ws.var_g0, ws.mean_sqr_g1, ws.var_g1);
    }
    if (l < stats.def_stats.bbvi_stats_z.size()) {
      auto & zs = stats.def_stats.bbvi_stats_z[l];
      printf("\tmean_sqr(g_z0) %.3e, mean_var(g_z0) %.3e, mean_sqr(g_z1) %.3e, mean_var(g_z1) %.3e\n",
             zs.mean_sqr_g0, zs.var_g0, zs.mean_sqr_g1, zs.var_g1);
    }
  }
}

void DEFModel::log_stats(const TrainStats& stats, ofstream& of) {
  stats.fill_for_print(model_type);

  char buffer[100];
  (void)buffer;
  auto d2s = [&](double d) {
    sprintf(buffer, "%.3e", d);
    return string(buffer);
  };
  vector<string> cols = { to_string(stats.iteration),
                          to_string((int)time(NULL)),
                          d2s(arma::mean(stats.def_stats.elbo)),
                          d2s(arma::mean(stats.lp_y)) };

  if (model_type != "train") {
    vector<string> pred_stats = stats.prediction_stats->vals();
    for (size_t l = 0; l < pred_stats.size(); ++l) {
      cols.push_back(pred_stats[l]);
    }
  }

  auto append_bbvi = [&](const BBVIStats& s) {
    LOG(trace) << "append bbvi info, len(cols)=" << cols.size();
    cols.push_back(d2s(s.mean_sqr_g0));
    cols.push_back(d2s(s.var_g0));
    cols.push_back(d2s(s.mean_sqr_g1));
    cols.push_back(d2s(s.var_g1));
    LOG(trace) << "after appending bbvi info, len(cols)=" << cols.size();
  };
  BBVIStats empty_stats;
  for(size_t l=0; l<def.get_layer_sizes().size(); ++l) {
    cols.push_back(d2s(arma::mean(stats.def_stats.lp_z[l])));
    cols.push_back(d2s(arma::mean(*stats.lp_w_for_print[l])));
    cols.push_back(d2s(arma::mean(stats.def_stats.lq_z[l])));
    cols.push_back(d2s(arma::mean(*stats.lq_w_for_print[l])));
    if (l < stats.w_for_print.size())
      append_bbvi(*stats.w_for_print[l]);
    else
      append_bbvi(empty_stats);
    if (l < stats.def_stats.bbvi_stats_z.size())
      append_bbvi(stats.def_stats.bbvi_stats_z[l]);
    else
      append_bbvi(empty_stats);
  }
  for (size_t i=0; i<cols.size(); ++i) {
    of << cols[i].c_str() << (i+1<cols.size() ? "\t" : "\n");
  }
  LOG(trace) << "len(cols)=" << cols.size();
  of.flush();
}

void DEFModel::log_stats_header(ofstream& of, const vector<string>& prediction_header) {
  vector<string> headers = {"iteration", "timestamp",
                            "elbo", "lp_y"};

  if (model_type != "train") {
    for (size_t l =0; l < prediction_header.size(); ++l) {
      headers.push_back(prediction_header[l]);
    }
  }

  for(size_t l=0; l<def.get_layer_sizes().size(); ++l) {
    headers.push_back("lp_z_" + to_string(l));
    headers.push_back("lp_w_" + to_string(l));
    headers.push_back("lq_z_" + to_string(l));
    headers.push_back("lq_w_" + to_string(l));
    headers.push_back("bbvi_sqr_gw0_" + to_string(l));
    headers.push_back("bbvi_var_gw0_" + to_string(l));
    headers.push_back("bbvi_sqr_gw1_" + to_string(l));
    headers.push_back("bbvi_var_gw1_" + to_string(l));
    headers.push_back("bbvi_sqr_gz0_" + to_string(l));
    headers.push_back("bbvi_var_gz0_" + to_string(l));
    headers.push_back("bbvi_sqr_gz1_" + to_string(l));
    headers.push_back("bbvi_var_gz1_" + to_string(l));
  }
  for (size_t i=0; i<headers.size(); ++i) {
    of << headers[i].c_str() << (i+1<headers.size() ? "\t" : "\n");
  }
  of.flush();
}

void DEFModel::init() {
  auto seed = ptree.get<int>("seed");
  n_samples = ptree.get<int>("samples");
  vec_rng.resize(n_samples);
  for(int i=0; i<n_samples; ++i) {
    vec_rng[i] = new GSLRandom();
    gsl_rng_set(vec_rng[i]->rng, seed+i);
  }
  iteration = 0;
  layers = 0;
  batch_st = 0;
  exp_fam_mode = ptree.get<bool>("exp_fam_mode");

  // data
  model_type = ptree.get<string>("model_type");

  if (data_file == "") {
    data_file = ptree.get<string>(model_type + ".data_file");
  }

  auto data_type = ptree.get<string>(model_type + ".data_type");
  def_data = build_def_data(data_type, ptree, data_file);
  if (ptree.get("global.data_transform", "id") == "log") {
    LOG(debug) << "transform data " << data_file << " with log";
    def_data->transform([](double x) {return int(round(log(1+x))); });
  }

  n_examples = def_data->n_examples();
  n_dim_y = def_data->n_dim_y();

  LOG(debug) << "n_examples " << n_examples;

  gsl_rng* rng = vec_rng[0]->rng;

  DEFInitializer initializer;
  initializer.rng = rng;
  initializer.def_data = def_data;

  // y
  {
    auto options = ptree;
    options.add_child("layer", ptree.get_child("y_layer"));
    LOG(debug) << "ptree y_layer.type=" << ptree.get<string>("y_layer.type");
    auto y_type = options.get<string>("layer.type");
    y_layer = GET_P_Y_LAYER(y_type, options, initializer);
  }

  // q(w) layer
  auto build_qw_layer = [&](int z_lower, int z_higher) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child("obs_weights"));
    options.put("n_examples", z_higher);
    options.put("layer.size", z_lower);
    options.put("rng", STORE_OBJ(vec_rng[0]));
    options.put("lf", options.get<string>("layer.q_lf"));
    auto layer_type = options.get<string>("layer.type");
    return GET_Q_LAYER(layer_type, options, initializer);
  };

  def.init_with_params(ptree, "DEF", def_data);

  int z_lower = n_dim_y, z_higher = def.get_layer_sizes()[0];
  q_w_obs_layer = build_qw_layer(z_lower, z_higher);
  if (ptree.get_child("obs_weights").get<string>("type") == "exp") {
    q_w_obs_layer_b = build_qw_layer(z_lower, z_higher);
  }
  else {
    q_w_obs_layer_b = NULL;
  }
  n_dim_z_1 = z_higher;

  auto build_prior_layer = [&](const string& layer_name) {
    pt::ptree options = ptree;
    options.add_child("layer", ptree.get_child(layer_name));
    auto layer_type = options.get<string>("layer.type");
    return GET_PRIOR_LAYER(layer_type, options, initializer);
  };
  // sparse prior distribution for obs_weight
  w_obs_layer = build_prior_layer("DEF_weights");
  if (ptree.get_child("DEF_weights").get<string>("type") == "exp") {
    w_obs_layer = build_prior_layer("DEF_weights");
  }
  else {
    w_obs_layer_b = NULL;
  }

  // setup logging, create some stats to get acess to generic header
  log_file.reset(new ofstream((get_output_folder() + "/" + model_type + ".log").c_str()));
  vector<string> prediction_header = y_layer->prediction_header();
  log_stats_header(*log_file, prediction_header);
}

static ExampleIds gen_example_ids(gsl_rng* rng, const string& batch_order,
				  int batch_size, int n_examples, int* batch_st) {

  (*batch_st) %= n_examples;
  ExampleIds examples;
  for (int j=0; j<batch_size; ++j)  {
    if (batch_order == "seq") {
      examples.push_back((*batch_st));
      ++(*batch_st);
      (*batch_st) %= n_examples;
    }
    else {
      (*batch_st) = gsl_rng_get(rng) % n_examples;
      examples.push_back(*batch_st);
    }
  }
  return examples;
}

void
DEFModel::train_model() {
  auto batch_size = ptree.get<int>("batch");
  auto batch_order = ptree.get<string>("batch_order");
  if (batch_size < 0) {
    batch_size = n_examples;
    batch_order = "seq";
  }

  batch_st = gsl_rng_get(vec_rng[0]->rng) % n_examples;
  ExampleIds ex = gen_example_ids(vec_rng[0]->rng, batch_order, batch_size, n_examples, &batch_size);
  auto stats = train_batch(ex);

  print_stats(stats);
  log_stats(stats, *log_file);
}

// compute log likelihood for validation/test
DEFModel::TrainStats
DEFModel::compute_log_likelihood() {
  shared_ptr<arma::mat> w_mean, z_mean, lp_mat;
  w_mean = q_w_obs_layer->mean_matrix();
  z_mean = def.mean();

  // iteration -1, layers 1, samples 1
  TrainStats stats(-1, 1, 1);
  stats.prediction_stats = y_layer->prediction_stats(w_mean, z_mean);
  return stats;
}

// perform one iteration of training on the examples
DEFModel::TrainStats
DEFModel::train_batch(const ExampleIds& example_ids) {

  auto samples = n_samples;
  q_w_obs_layer->truncate();

  // samples;
  VecOfMat w_obs_samples;
  w_obs_samples.resize(samples);

  vector<shared_ptr<arma::rowvec> > samples_lpy_row;
  samples_lpy_row.resize(samples);
  vector<shared_ptr<arma::colvec> > samples_lpy_col;
  samples_lpy_col.resize(samples);

  VecOfMat samples_log_pw, samples_log_qw;
  samples_log_pw.resize(samples);
  samples_log_qw.resize(samples);

  VecOfCube samples_score_qw;
  samples_score_qw.resize(samples);

  TrainStats stats(iteration++, def.num_layers(), samples);
  if (model_type != "train") {
    assert(ptree.get<int>("batch") == -1);
    auto test_stats = compute_log_likelihood();
    stats.prediction_stats = test_stats.prediction_stats;
  }

  auto t0 = time(NULL);

  def.prepare_to_sample(example_ids);

  auto threads = ptree.get<int>("threads");
#pragma omp parallel for num_threads(threads), schedule(dynamic, samples/threads)
  for(int s=0; s<samples; ++s) {
    gsl_rng* rng = vec_rng[s]->rng;

    w_obs_samples[s] = q_w_obs_layer->sample_matrix(rng);
    samples_log_pw[s] = w_obs_layer->log_p_matrix(w_obs_samples[s]);
    samples_log_qw[s] = q_w_obs_layer->log_q_matrix(w_obs_samples[s]);
    samples_score_qw[s] = q_w_obs_layer->grad_lq_matrix(w_obs_samples[s]);

    // doesn't support double-exponential distribution for obs_weights currently
    assert((w_obs_layer_b == NULL) && (q_w_obs_layer_b == NULL));

    // Sample the DEF, pass stats?
    shared_ptr<arma::mat> z_1_samples = def.sample(example_ids, s, rng, &stats.def_stats);

    // log p(y)
    auto sampling_ratio = -1.0;
    auto res =
      y_layer->log_p_row_column(w_obs_samples[s], z_1_samples, example_ids);
    samples_lpy_row[s] = res.log_p_row_train;
    samples_lpy_col[s] = res.log_p_col_train;
    stats.lp_y(s) = arma::sum(*res.log_p_row_train);
    sampling_ratio = (example_ids.size()+0.0) / n_examples;

    // renormalize log p(w) & log q(w)
    *samples_log_pw[s] *= sampling_ratio;
    *samples_log_qw[s] *= sampling_ratio;

    // compute the statistics for logging
    stats.def_stats.elbo(s) += stats.lp_y(s);
    stats.def_stats.elbo(s) += (stats.lp_w_obs(s) = arma::accu(*samples_log_pw[s]));
    stats.def_stats.elbo(s) -= (stats.lq_w_obs(s) = arma::accu(*samples_log_qw[s]));
  }

  auto t1 = time(NULL);
  LOG(debug) << "sampling takes " << t1-t0 << " seconds";

  LOG(trace) << "computing bbvi";
  if (model_type == "train") {
    VecOfMat lp_list;
    lp_list.resize(n_samples);
    auto threads = ptree.get<int>("threads");
    #pragma omp parallel for num_threads(threads)
    for (int s = 0; s < n_samples; ++s) {
      shared_ptr<arma::mat> lp( new arma::mat(n_dim_y, n_dim_z_1) );
      lp->each_col() = *samples_lpy_col[s];
      *lp += *samples_log_pw[s];
      lp_list[s] = lp;
    }
    stats.w_obs_layer = q_w_obs_layer->update(samples_score_qw, lp_list, samples_log_qw);
  }

  def.update(example_ids, samples_lpy_row, &stats.def_stats);

  auto t2 = time(NULL);
  LOG(debug) << "bbvi takes " << t2-t1 << " seconds";
  return stats;
}

void DEFModel::save_params(const string& fname) const {
  FILE* ofile = fopen(fname.c_str(), "wb");
  LOG(debug) << "model save params to " << fname;
  q_w_obs_layer->save_params(ofile);
  def.save_params(ofile);
  fclose(ofile);
}

void DEFModel::load_params(const string& fname) {
  FILE* ifile = fopen(fname.c_str(), "rb");
  LOG(debug) << "model load params from " << fname;
  q_w_obs_layer->load_params(ifile);
  def.load_params(ifile);
  fclose(ifile);
}

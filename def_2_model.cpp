#include <boost/optional/optional.hpp>
#include "def_2_model.hpp"
#include "layer_factory.hpp"

using boost::optional;

static void print_DEF_stats(const DEF::TrainStats& stats, const string& prefix, size_t n_layers) {
  size_t l;
  for(l=0; l<n_layers - 1; ++l) {
    printf("\t%s Layer %d, lp_z %.3e, lp_w %.3e, lq_z %.3e, lq_w %.3e\n", prefix.c_str(), (int)l,
	   arma::mean(stats.lp_z[l]), arma::mean(stats.lp_w[l]),
	   arma::mean(stats.lq_z[l]), arma::mean(stats.lp_w[l]));
    if (l < stats.bbvi_stats_w.size()) {
      auto & ws = stats.bbvi_stats_w[l];
      printf("\tmean_sqr(g_w0) %.3e, mean_var(g_w0) %.3e, mean_sqr(g_w1) %.3e, mean_var(g_w1) %.3e\n",
	     ws.mean_sqr_g0, ws.var_g0, ws.mean_sqr_g1, ws.var_g1);
    }
    if (l < stats.bbvi_stats_z.size()) {
      auto & zs = stats.bbvi_stats_z[l];
      printf("\tmean_sqr(g_z0) %.3e, mean_var(g_z0) %.3e, mean_sqr(g_z1) %.3e, mean_var(g_z1) %.3e\n",
	     zs.mean_sqr_g0, zs.var_g0, zs.mean_sqr_g1, zs.var_g1);
    }
  }

  printf("\t%s Final z, lp_z %.3e, lq_z %.3e", prefix.c_str(), arma::mean(stats.lp_z[l]), arma::mean(stats.lq_z[l]));
  auto & zs = stats.bbvi_stats_z[l];
  printf("\tmean_sqr(g_z0) %.3e, mean_var(g_z0) %.3e, mean_sqr(g_z1) %.3e, mean_var(g_z1) %.3e\n",
	 zs.mean_sqr_g0, zs.var_g0, zs.mean_sqr_g1, zs.var_g1);
}

void DEF2Model::print_stats(const TrainStats& stats) {
  printf("Iteration %d, ELBO %.3e, std %.3e, lp(y) %.3e\n",
         stats.iteration, arma::mean(stats.elbo), arma::stddev(stats.elbo),
         arma::mean(stats.lp_y));
  if (model_type != "train") {
    printf("\t%s_metric: ", model_type.c_str());
    stats.prediction_stats->pretty_print();
  }
  else { 
    print_DEF_stats(stats.col_stats, "COL", def_cols.num_layers());
  }
  print_DEF_stats(stats.row_stats, "ROW", def_rows.num_layers());
}

static string d2s(double d) {
  char buffer[100];

  (void)buffer;
  sprintf(buffer, "%.3e", d);
  return string(buffer);
}

static void add_stats(vector<string>* cols, const DEF::TrainStats& def_stats, const DEF& def) {
  auto append_bbvi = [&](const BBVIStats& s) {
    LOG(trace) << "append bbvi info, len(cols)=" << cols->size();
    cols->push_back(d2s(s.mean_sqr_g0));
    cols->push_back(d2s(s.var_g0));
    cols->push_back(d2s(s.mean_sqr_g1));
    cols->push_back(d2s(s.var_g1));
    LOG(trace) << "after appending bbvi info, len(cols)=" << cols->size();
  };
  BBVIStats empty_stats;
  for(size_t l=0; l < def.get_layer_sizes().size(); ++l) {
    cols->push_back(d2s(arma::mean(def_stats.lp_z[l])));
    cols->push_back(d2s(arma::mean(def_stats.lq_z[l])));
    if (l < def_stats.bbvi_stats_z.size())
      append_bbvi(def_stats.bbvi_stats_z[l]);
    else
      append_bbvi(empty_stats);
    
    if (l < def.num_layers() - 1) {
      cols->push_back(d2s(arma::mean(def_stats.lq_w[l])));
      cols->push_back(d2s(arma::mean(def_stats.lp_w[l])));
      if (l < def_stats.bbvi_stats_w.size())
	append_bbvi(def_stats.bbvi_stats_w[l]);
      else
	append_bbvi(empty_stats);
    }
  }
}

void DEF2Model::log_stats(const TrainStats& stats, ofstream& of) {

  vector<string> cols = { to_string(stats.iteration),
                          to_string((int)time(NULL)),
                          d2s(arma::mean(stats.elbo)),
                          d2s(arma::mean(stats.lp_y)) };

  if (model_type != "train") {
    vector<string> pred_stats = stats.prediction_stats->vals();
    for (size_t l = 0; l < pred_stats.size(); ++l) {
      cols.push_back(pred_stats[l]);
    }
  }

  add_stats(&cols, stats.row_stats, def_rows);
  add_stats(&cols, stats.col_stats, def_cols);

  for (size_t i=0; i<cols.size(); ++i) {
    of << cols[i].c_str() << (i+1<cols.size() ? "\t" : "\n");
  }
  LOG(trace) << "len(cols)=" << cols.size();
  of.flush();
}

static void add_log_header(vector<string>* headers, const DEF& def) {
  for(size_t l=0; l<def.num_layers(); ++l) {
    headers->push_back("lp_z_" + to_string(l));
    headers->push_back("lq_z_" + to_string(l));
    headers->push_back("bbvi_sqr_gz0_" + to_string(l));
    headers->push_back("bbvi_var_gz0_" + to_string(l));
    headers->push_back("bbvi_sqr_gz1_" + to_string(l));
    headers->push_back("bbvi_var_gz1_" + to_string(l));
    if (l < def.num_layers() - 1) {
      headers->push_back("lp_w_" + to_string(l));
      headers->push_back("lq_w_" + to_string(l));
      headers->push_back("bbvi_sqr_gw0_" + to_string(l));
      headers->push_back("bbvi_var_gw0_" + to_string(l));
      headers->push_back("bbvi_sqr_gw1_" + to_string(l));
      headers->push_back("bbvi_var_gw1_" + to_string(l));
    }
  }
}

void DEF2Model::log_stats_header(ofstream& of, const vector<string>& prediction_header) {
  vector<string> headers = {"iteration", "timestamp",
                            "elbo", "lp_y"};

  if (model_type != "train") {
    for (size_t l =0; l < prediction_header.size(); ++l) {
      headers.push_back(prediction_header[l]);
    }
  }
  add_log_header(&headers, def_rows);
  add_log_header(&headers, def_cols);
  for (size_t i=0; i<headers.size(); ++i) {
    of << headers[i].c_str() << (i+1<headers.size() ? "\t" : "\n");
  }
  of.flush();
}

void DEF2Model::init() {
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

  // data
  model_type = ptree.get<string>("model_type");

  if (data_file == "") {
    data_file = ptree.get<string>(model_type + ".data_file");
  }
  auto data_type = ptree.get<string>(model_type + ".data_type");
  def_data = build_def_data(data_type, ptree, data_file);
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

  def_rows.init_with_params(ptree, "DEF_rows", def_data);
  def_cols.init_with_params(ptree, "DEF_cols", def_data->transpose());

  for (arma::uword d_y = 0; d_y < n_dim_y; ++d_y) {
    full_col_ids.push_back(d_y);
  }

  // setup logging
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
DEF2Model::train_model() {
  auto batch_size = ptree.get<int>("batch");
  auto batch_order = ptree.get<string>("batch_order");
  if (batch_size < 0) {
    batch_size = n_examples;
    batch_order = "seq";
  }

  batch_st = gsl_rng_get(vec_rng[0]->rng) % n_examples;
  ExampleIds ex_rows = gen_example_ids(vec_rng[0]->rng, batch_order, batch_size, n_examples, &batch_size);
  auto stats = train_batch(ex_rows);

  print_stats(stats);
  log_stats(stats, *log_file);
}

DEF2Model::TrainStats
DEF2Model::compute_log_likelihood() {
  shared_ptr<arma::mat> z_r_mean, z_c_mean, lp_mat;
  z_r_mean = def_rows.mean();

  z_c_mean.reset(new arma::mat(def_cols.mean()->t()));

  TrainStats stats(-1, 1, 1, 1);
  stats.prediction_stats = y_layer->prediction_stats(z_c_mean, z_r_mean);
  return stats;
}

DEF2Model::TrainStats 
DEF2Model::train_batch(const ExampleIds& ex_rows) {
  auto samples = n_samples;
  vector<shared_ptr<arma::rowvec> > samples_lpy_row;
  samples_lpy_row.resize(samples);
  vector<shared_ptr<arma::rowvec> > samples_lpy_col;
  samples_lpy_col.resize(samples);

  TrainStats stats(iteration++, def_rows.num_layers(), def_cols.num_layers(), samples);
  if (model_type != "train") {
    assert(ptree.get<int>("batch") == -1);
    auto test_stats = compute_log_likelihood();
    stats.prediction_stats = test_stats.prediction_stats;
  }

  auto t0 = time(NULL);

  def_rows.prepare_to_sample(ex_rows);
  def_cols.prepare_to_sample(full_col_ids);

  auto threads = ptree.get<int>("threads");
  #pragma omp parallel for num_threads(threads), schedule(dynamic, samples/threads)
  for(int s=0; s<samples; ++s) {
    gsl_rng* rng = vec_rng[s]->rng;
    
    LOG(trace) << "sampling s=" << s;
    LOG(trace) << "sample rows";
    shared_ptr<arma::mat> z_row_samples = def_rows.sample(ex_rows, s, rng, &stats.row_stats); 
    shared_ptr<arma::mat> z_col_samples(new arma::mat(def_cols.sample(full_col_ids, s, rng, &stats.col_stats)->t()));
    
    LOG(trace) << "computing log p(y)";
    auto sampling_ratio = -1.0;
    auto res = y_layer->log_p_row_column(z_col_samples, z_row_samples, ex_rows);
    samples_lpy_row[s] = res.log_p_row_train;
    samples_lpy_col[s].reset(new arma::rowvec(res.log_p_col_train->t()));
    stats.lp_y(s) = arma::sum(*res.log_p_row_train);

    LOG(trace) << "computing stats";
    stats.elbo(s) = stats.lp_y(s) + stats.row_stats.elbo(s) + stats.col_stats.elbo(s);
  }

  auto t1 = time(NULL);
  LOG(debug) << "sampling takes " << t1-t0 << " seconds";

  LOG(trace) << "computing bbvi";
  def_rows.update(ex_rows, samples_lpy_row, &stats.row_stats);
  if (model_type == "train") {
    def_cols.update(full_col_ids, samples_lpy_col, &stats.col_stats);
  }

  // bbvi for L layers
  auto t2 = time(NULL);
  LOG(debug) << "bbvi takes " << t2-t1 << " seconds";
  return stats;
}

void DEF2Model::save_params(const string& fname) const {
  FILE* ofile = fopen(fname.c_str(), "wb");
  LOG(debug) << "model save params to " << fname;
  def_rows.save_params(ofile);
  def_cols.save_params(ofile);
  fclose(ofile);
}

void DEF2Model::load_params(const string& fname) {
  FILE* ifile = fopen(fname.c_str(), "rb");
  LOG(debug) << "model load params from " << fname;
  def_rows.load_params(ifile);
  def_cols.load_params(ifile);
  fclose(ifile);  
}

#include "def_data.hpp"

shared_ptr<DEFData> build_def_data(const string& data_type,
                                   const pt::ptree& options,
                                   const string& fname) {
  if (data_type == "sparse-text") {
    return shared_ptr<DEFData>(new SparseTextData(options, fname));
  }
  else if (data_type == "masked-text") {
    return shared_ptr<DEFData>(new MaskedTextData(options, fname));
  }
  else {
    throw runtime_error("unknown data type");
  }
}


shared_ptr<arma::sp_mat> read_text_data(const string& fname, int max_examples) {
  LOG(debug) << "reading text data from " << fname
             << " max_examples " << max_examples;
  ifstream fin(fname);
  int n_examples, n_dim;
  fin >> n_examples >> n_dim;
  vector<arma::uword> x_list, y_list;
  vector<double> z_list;
  for (int i=0; i<n_examples && ((max_examples<0) || (i<max_examples)); ++i) {
    int x, y, z, nnz;
    fin >> x >> nnz;
    for(int j=0; j<nnz; ++j) {
      fin >> y >> z; 
      x_list.push_back(x);
      y_list.push_back(y);
      z_list.push_back(z);
      assert(y < n_dim);
    }
  }
  // note we store the transposition of the data
  arma::umat locations = arma::join_cols(arma::umat(y_list).t(),
                                         arma::umat(x_list).t());
  arma::vec values(z_list);
  LOG(debug) << "locations " << "rows=" << locations.n_rows
             << " columns=" << locations.n_cols;
  LOG(debug) << "values " << "rows=" << values.n_rows
             << " columns=" << values.n_cols;    
  if (max_examples >= 0)
    n_examples = min(n_examples, max_examples);
  auto data = new arma::sp_mat(locations, values, n_dim, n_examples);
  LOG(debug) << "rows(data)=" << data->n_rows
             << " columns(data)=" << data->n_cols;
  return shared_ptr<arma::sp_mat>(data);
}

shared_ptr<DEFData> SparseTextData::transpose() const {
  SparseTextData* trans_data = new SparseTextData();
  trans_data->options = options;
  trans_data->data.reset(new arma::sp_mat(data->t()));
  return shared_ptr<DEFData>(trans_data);
}

MaskedTextData::MaskedTextData(const pt::ptree& options, const string& fname)
  : options(options){
  auto max_examples = options.get<int>("max_examples");
  auto exp_fname = expand_environment_variables(fname);
  arma::mat masked_data(*read_text_data(exp_fname, max_examples));
  test_filter = shared_ptr<arma::mat>( new arma::mat(masked_data) );
  test_filter->transform([](double v) { return v < 0 ? 1 : 0; });
  train_filter = shared_ptr<arma::mat>( new arma::mat(masked_data) );
  train_filter->transform([](double v) { return v >= 0 ? 1 : 0; });

  masked_data.transform([](double v) { return v >= 0 ? v : -v-1; });
  data = shared_ptr<arma::sp_mat>( new arma::sp_mat(masked_data) );
  LOG(debug) << "test data nnz=" << data->n_nonzero;
}

shared_ptr<DEFData> MaskedTextData::transpose() const {
  MaskedTextData* trans_data =new MaskedTextData();
  trans_data->options = options;
  trans_data->data.reset(new arma::sp_mat(data->t()));
  trans_data->test_filter.reset(new arma::mat(test_filter->t()));
  trans_data->train_filter.reset(new arma::mat(train_filter->t()));
  return shared_ptr<DEFData>(trans_data);
}

DenseData::DenseData(const pt::ptree& options, const string& fname) {
  ifstream fin(fname);
  arma::uword n_rows, n_cols;
  fin >> n_rows >> n_cols;
  for(arma::uword i=0; i<n_rows; ++i) {
    for(arma::uword j=0; j<n_cols; ++j) {
      // TODO fin >> 
    }
  }
}

shared_ptr<DEFData> DenseData::transpose() const {
  DenseData* trans_data = new DenseData();
  trans_data->options = options;
  trans_data->data.reset(new arma::mat(data->t()));
  return shared_ptr<DEFData>(trans_data);
}

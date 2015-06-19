#include <regex>
#include <sstream>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include "utils.hpp"

logging::trivial::severity_level severityLevel;

void init_logging() {
  // initial logging set up
  logging::core::get()->set_filter
    (
     logging::trivial::severity >= logging::trivial::info
     );
}

void init_logging(const pt::ptree& ptree) {
  LOG(debug) << "v=" << ptree.get<int>("v");
  switch (ptree.get<int>("v"))
    {
    case 4:
      severityLevel = logging::trivial::trace;
      break;
    case 3:
      severityLevel = logging::trivial::debug;
      break;
    case 2:
      severityLevel = logging::trivial::info;
      break;
    case 1:
      severityLevel = logging::trivial::warning;
      break;
    default:
        throw runtime_error("unknown logging level");
    }
  logging::core::get()->set_filter(logging::trivial::severity >= severityLevel);
}

map<string, any> object_map;
int object_map_counter;

string gen_random_string64() {
  string chars("abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  boost::random::random_device rng;
  boost::random::uniform_int_distribution<> index_dist(0, chars.size() - 1);
  stringstream ss;
  for(int i = 0; i < 64; ++i) {
    ss << chars[index_dist(rng)];
  }
  return ss.str();
}

std::string expand_environment_variables( std::string s ) {
  LOG(debug) << "expanding " << s;
  if( s.find( "${" ) == std::string::npos ) return s;

  std::string pre  = s.substr( 0, s.find( "${" ) );
  std::string post = s.substr( s.find( "${" ) + 2 );

  if( post.find( '}' ) == std::string::npos ) return s;

  std::string variable = post.substr( 0, post.find( '}' ) );
  std::string value    = "";

  post = post.substr( post.find( '}' ) + 1 );

  if( getenv( variable.c_str() ) != NULL ) value = std::string( getenv( variable.c_str() ) );
  LOG(trace) << "try to substitute " << variable << " with " << value;

  return expand_environment_variables( pre + value + post );
}

// slice columns from a matrix
shared_ptr<arma::sp_mat> slice_cols(shared_ptr<arma::sp_mat> x,
                                    const ExampleIds& col_ids) {
  vector<arma::uword> x_list, y_list;
  vector<double> z_list;
  for(size_t j=0; j<col_ids.size(); ++j) {
    auto col = x->col(col_ids[j]);
    int i_last = -1;
    for(auto it=col.begin(); it != col.end(); ++it) {
    // for(auto it=x->begin_col(col_ids[j]);
    //     it != x->end_col(col_ids[j]); ++it) {
      auto i = it.row();
      auto v = *it;
      LOG_IF(fatal, (int) i <= i_last) << "i=" << i << " i_last=" << i_last
                                       << " j=" << j << " col_ids[j]=" << col_ids[j];
      assert((int) i > i_last);
      i_last = i;
      //LOG(trace) << "i=" << i << " j=" << j << " v=" << v;
      x_list.push_back(i);
      y_list.push_back(j);
      z_list.push_back(v);
    }
  }
  // note we store the transposition of the data
  arma::umat locations = arma::join_cols(arma::umat(x_list).t(),
                                         arma::umat(y_list).t());
  LOG(trace) << "locations\n" << locations.cols(0, 4);
  arma::vec values(z_list);
  LOG(trace) << "values\n" << values.rows(0, 4);

  auto data = new arma::sp_mat(locations, values, x->n_rows, col_ids.size());
  LOG(debug) << "slicing sp_mat, rows=" << data->n_rows
             << " columns=" << data->n_cols;
  return shared_ptr<arma::sp_mat>(data);
}


shared_ptr<arma::mat> slice_cols(shared_ptr<arma::mat> x,
                                 const ExampleIds& col_ids) {
  return shared_ptr<arma::mat>( new arma::mat(x->cols(arma::uvec(col_ids))) );
}

static string output_folder;
string get_output_folder() {
  return output_folder;
}

void set_output_folder(const string& folder) {
  output_folder = folder;
}

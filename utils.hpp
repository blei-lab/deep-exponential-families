// utility functions, included everywhere
#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <fstream>

#include <boost/any.hpp>
#include <boost/program_options.hpp> 
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree_serialization.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

#define ARMA_64BIT_WORD
#include <armadillo>
#include "serialization.hpp"

using namespace std;
typedef vector<shared_ptr<arma::rowvec> > VecOfRow;
typedef vector<std::shared_ptr<arma::mat> > VecOfMat;
typedef vector<std::shared_ptr<arma::cube> > VecOfCube;
typedef vector<arma::uword> ExampleIds;

using boost::any;
using boost::any_cast;
namespace po = boost::program_options;
namespace pt = boost::property_tree;
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

extern logging::trivial::severity_level severityLevel;

#define LOG(L) BOOST_LOG_TRIVIAL(L) <<  __FILE__ << ":" << __LINE__ << " "
#define LOG_IS_ON(L) (severityLevel <= logging::trivial::L)
#define LOG_IF(L, COND) if (COND) \
    BOOST_LOG_TRIVIAL(L) <<  __FILE__ << ":" << __LINE__ << " "

// called before parsing options
void init_logging();

// called after parsing options
void init_logging(const pt::ptree& ptree);


// ---- Bernoulli related ----
template<typename T>
T softmax(T x) {
  if (x > 10)
    return x;
  else if (x < -10)
    return exp(x);
  else
    return log(1+exp(x));
}

template<typename T>
T sigmoid(T x) {
  return 1.0/(1.0+exp(-x));
}

// multiply w by some columns of x
template<typename T>
shared_ptr<arma::mat> mat_mul_cols(const arma::mat& w, const T& x, const ExampleIds& col_ids) {
  shared_ptr<arma::mat> res( new arma::mat(w.n_rows, col_ids.size()) );
  for(size_t j=0; j<col_ids.size(); ++j)
    res->col(j) = w * x.col(col_ids[j]);
  return res;
}

// return matrix a - b
template<typename T>
shared_ptr<arma::mat> arma_sub(const T& a, const T& b) {
  shared_ptr<arma::mat> res( new arma::mat(a.n_rows, a.n_cols) );
  *res = a;
  *res -= b;
  return res;
}

// slice columns from a matrix
shared_ptr<arma::sp_mat> slice_cols(shared_ptr<arma::sp_mat> x, const ExampleIds& col_ids);
shared_ptr<arma::mat> slice_cols(shared_ptr<arma::mat> x, const ExampleIds& col_ids);

template<typename ValueTy>
void set_vm(po::variables_map& m, const string& name, const ValueTy& v){
  if (m.count(name)) {
    po::variables_map::iterator it(m.find(name));
    po::variable_value & vx(it->second); 
    vx.value() = v; 
  }
  else {
    m.insert( make_pair(name, po::variable_value(v, false)) );
  }
}

// handle --> objects
extern map<string, any> object_map;
extern int object_map_counter;
string gen_random_string64();

// store a object and return a handle
template<typename T>
string STORE_OBJ(const T& v) {
  // auto handle = gen_random_string64();
  auto handle = "handle_" + to_string(object_map_counter++);
  assert(object_map.count(handle) == 0);
  object_map[handle] = any(v);
  return handle;
}

template<typename T>
T GET_OBJ(const string& handle) {
  assert(object_map.count(handle) > 0);
  return any_cast<T>(object_map[handle]);
}

// write a binary array to file
template<typename T>
void save_mat(FILE* ofile, const T& W) {
  size_t r = W.n_rows;
  size_t c = W.n_cols;
  double* data=new double[r*c+2];
  data[0] = r;
  data[1] = c;
  size_t it = 2;
  // due to padding, this is the safe way to get the continugous data
  for(size_t i=0; i<r; ++i) {
    for(size_t j=0; j<c; ++j) {
      data[it++] = W(i,j);
    }
  }
  fwrite((void *) data, sizeof(char),
         (r*c+2)*sizeof(double), ofile);
  delete[] data;
}

// load a binary array from file
template<typename T>
void load_mat(FILE* ifile, T& W) {
  double rc_buffer[2];
  fread((char*)rc_buffer, sizeof(double), 2, ifile);
  arma::uword r = (arma::uword) rc_buffer[0];
  arma::uword c = (arma::uword) rc_buffer[1];
  LOG(debug) << "read array " << "row(W)=" << W.n_rows << " col(W)=" << W.n_cols
             << " row(array)=" << r << " col(array)=" << c;
  assert((r == W.n_rows) && (c == W.n_cols));

  double* data=new double[r*c];
  fread(data, sizeof(double), r*c, ifile);
  size_t it = 0;
  // due to padding, this is the safe way to get the continugous data
  for(arma::uword i=0; i<r; ++i) {
    for(arma::uword j=0; j<c; ++j) {
      W(i,j) = data[it++];
    }
  }
  LOG(trace) << "load mat\n" << W.submat(0, 0, 3, 3);
  delete[] data;
}

// expand ${..} environment variables
string expand_environment_variables( std::string s );

// The current output 
string get_output_folder();
void set_output_folder(const string& folder);

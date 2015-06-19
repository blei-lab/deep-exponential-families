#ifndef DEF_LAYER_HPP
#define DEF_LAYER_HPP

#include <gsl/gsl_rng.h>
#include "bbvi.hpp"
#include "optimizer.hpp"
#include "optimizer.hpp"
#include "serialization.hpp"
#include "def_data.hpp"

// stuff that can't be put to the property tree
// but useful for initialization
struct DEFInitializer {
  gsl_rng* rng;
  shared_ptr<DEFData> def_data;
  DEFInitializer() {
  }

private:
  // make the copy constructor and the = private to prevent serialization
  DEFInitializer(const DEFInitializer& other) {
  }
  DEFInitializer& operator=(DEFInitializer& other) {
    return *this;
  }
};

class DEFLayer {
public:

  virtual double compute_log_p(double z, double param) = 0;


  // TODO: Clean up these similar functions
  shared_ptr<arma::mat> log_p_matrix(shared_ptr<arma::mat> w,
                                     shared_ptr<arma::mat> z,
                                     shared_ptr<arma::mat> z_higher,
                                     shared_ptr<arma::mat> z_bias) {
    arma::mat param = (*w) * (*z_higher);
    if (z_bias != NULL) {
      assert(z_bias->n_cols == 1);
      param.each_col() += z_bias->col(0);
    }
    shared_ptr<arma::mat> log_p( new arma::mat(z->n_rows, z->n_cols) );
    for(arma::uword j=0; j<z->n_cols; ++j) {
      for(arma::uword i=0; i<z->n_rows; ++i) {
        // TODO inline?
        (*log_p)(i, j) = compute_log_p((*z)(i, j), param(i, j));
      }
    }
    return log_p;
  }  
};

class DEFPriorLayer {
public:
  virtual double compute_log_p(double z) = 0;

  shared_ptr<arma::mat> log_p_matrix(shared_ptr<arma::mat> z) {
    shared_ptr<arma::mat> log_p( new arma::mat(z->n_rows, z->n_cols) );
    for(arma::uword j=0; j<z->n_cols; ++j) {
      for(arma::uword i=0; i<z->n_rows; ++i) {
        (*log_p)(i, j) = compute_log_p((*z)(i, j));
      }
    }
    return log_p;    
  }
};

class InferenceFactorizedLayer {
public:
  typedef function<double(double z, arma::uword i, arma::uword j)> ScoreFunction;

private:
  vector<ScoreFunction> score_funcs;
  vector<Serializable<arma::mat>* > param_matrices;
  vector<Optimizer> optimizers;

protected:
  int threads;
  const pt::ptree options;
  arma::uword n_examples;
  ExampleIds all_examples;

public:
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & (*const_cast<pt::ptree*>(&options));
    init();
    ar & optimizers;
    ar & param_matrices;
  }


  InferenceFactorizedLayer() {
  }

  InferenceFactorizedLayer(const pt::ptree& options)
    : options( options ) {
    init();
  }

  void init() {
    n_examples = options.get<int>("n_examples");
    all_examples.clear();
    for(arma::uword j=0; j<n_examples; ++j)
      all_examples.push_back(j);
    threads = options.get<int>("threads");  
  }

  virtual ~InferenceFactorizedLayer() {}

  virtual double compute_log_q(double z, arma::uword i, arma::uword j) = 0;
  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) = 0;
  virtual double mean(arma::uword i, arma::uword j) = 0;
  virtual void copy_params(InferenceFactorizedLayer* other) = 0;

  // truncate the parameters
  virtual void truncate(const ExampleIds& example_ids) = 0;
  virtual void truncate() = 0;

  void register_param(Serializable<arma::mat>* param_mat, ScoreFunction score_func, bool deserialize) {
    if (!deserialize) {
      assert(param_mat->n_cols == n_examples);
      optimizers.emplace_back(options, param_mat);
      param_matrices.push_back(param_mat);
    }
    score_funcs.push_back(score_func);
  }
  
  // z: layer-size x num-examples
  shared_ptr<arma::cube> grad_lq_matrix(shared_ptr<arma::mat> z,
                                        const ExampleIds& example_ids) {
    size_t n_params = param_matrices.size();
    shared_ptr<arma::cube> grad_lp( new arma::cube(z->n_rows, z->n_cols, n_params) );
    arma::uword ind = 0;
    for(auto j : example_ids) {
      for(arma::uword i=0; i<z->n_rows; ++i) {
        for(size_t k=0; k<score_funcs.size(); ++k) {
          (*grad_lp)(i, ind, k) = score_funcs[k]((*z)(i,ind), i, j);
        }
      }
      ++ind;
    }
    return grad_lp;
  }

  shared_ptr<arma::cube> grad_lq_matrix(shared_ptr<arma::mat> z) {
    return grad_lq_matrix(z, all_examples);
  }

  shared_ptr<arma::mat> log_q_matrix(shared_ptr<arma::mat> z,
                                        const ExampleIds& example_ids) {
    shared_ptr<arma::mat> log_q( new arma::mat(z->n_rows, z->n_cols) );
    arma::uword ind = 0;
    for(auto j : example_ids) {
      for(arma::uword i=0; i<z->n_rows; ++i) {
        // TODO inline?
        (*log_q)(i, ind) = compute_log_q((*z)(i, ind), i, j);
      }
      ++ind;
    }
    return log_q;
  }
  shared_ptr<arma::mat> log_q_matrix(shared_ptr<arma::mat> z) {
    return log_q_matrix(z, all_examples);
  }

  shared_ptr<arma::mat> sample_matrix(gsl_rng* rng, const ExampleIds& example_ids) {
    auto n_rows = param_matrices[0]->n_rows;
    shared_ptr<arma::mat> sample_mat( new arma::mat(n_rows, example_ids.size()) );
    arma::uword ind = 0;
    for(auto j : example_ids) {
      for(arma::uword i=0; i<n_rows; ++i) {
        // TODO inline?
        (*sample_mat)(i, ind) = sample(rng, i, j);
      }
      ++ind;
    }
    return sample_mat;
  }

  shared_ptr<arma::mat> sample_matrix(gsl_rng* rng) {
    return sample_matrix(rng, all_examples);
  }

  shared_ptr<arma::mat> mean_matrix(const ExampleIds& example_ids) {
    auto n_rows = param_matrices[0]->n_rows;
    auto n_cols = example_ids.size();
    shared_ptr<arma::mat> mean_mat( new arma::mat(n_rows, n_cols) );
    arma::uword ind = 0;
    for(auto j : example_ids) {
      for(arma::uword i=0; i<n_rows; ++i) {
        // TODO inline?
        (*mean_mat)(i, ind) = mean(i, j);
      }
      ++ind;
    }
    return mean_mat;
  }

  shared_ptr<arma::mat> mean_matrix() {
    return mean_matrix(all_examples);
  }

  BBVIStats update(const VecOfCube& score_q,
              const VecOfMat& log_p,
              const VecOfMat& log_q,
              const ExampleIds& example_ids) {
    BBVIStats stats;
    auto n_params = param_matrices.size();
    for(arma::uword k=0; k<n_params; ++k) {      
      // This is inefficent just pass and index to bbvi
      VecOfMat score_k;
      for(auto c : score_q)
        score_k.emplace_back( new arma::mat(c->slice(k)) );

      BBVIStats stats_k;
      auto grad_k = grad_bbvi_factorized(options, score_k, log_p, log_q, stats_k, threads);
      stats += stats_k;
      optimizers[k].update(*grad_k, example_ids);
    }
    stats /= (n_params+0.0);
    return stats;
  }

  BBVIStats update(const VecOfCube& score_q,
              const VecOfMat& log_p,
              const VecOfMat& log_q) {
    return update(score_q, log_p, log_q, all_examples);
  }

  // save params in binary format, save at least max_examples
  // columns for each parameter matrix
  void save_params(FILE* ofile, int max_examples) {
    if (max_examples > (int) n_examples)
      max_examples = n_examples;
    //cout << " max_examples" << max_examples << endl;
    for(auto w : param_matrices) {
      arma::mat sub_w = w->cols(0, max_examples-1);
      LOG(trace) << "save one param matrix";
      save_mat(ofile, sub_w);
    }
  }

  void save_params(FILE* ofile) {
    save_params(ofile, n_examples);
  }

  // load params
  void load_params(FILE* ifile, int max_examples) {
    if ((max_examples < 0) || (max_examples > (int) n_examples))
      max_examples = n_examples;
    for(auto w : param_matrices) {
      // we only load the whole matrics here
      LOG(trace) << "load params " << "max_examples=" << max_examples
                 << " cols(w)=" << w->n_cols;
      assert(max_examples == (int) w->n_cols);
      LOG(trace) << "load one param matrix";
      load_mat(ifile, *w);
    }
  }

  void load_params(FILE* ifile) {
    load_params(ifile, n_examples);
  }

  // check the parameter matrix
  void check_params(int m=4) {
    for (size_t k=0; k<param_matrices.size(); ++k) {
      LOG(debug) << "param matrix " << k << "\n"
                 << param_matrices[k]->submat(0, 0, m-1, m-1);
    }
  }
};
BOOST_SERIALIZATION_ASSUME_ABSTRACT(InferenceFactorizedLayer)

#endif

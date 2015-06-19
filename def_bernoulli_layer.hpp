#pragma once

#include <cassert>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include "utils.hpp"
#include "def_layer.hpp"
#include "link_function.hpp"
#include "serialization.hpp"

class BernoulliLayer : public DEFLayer {

private:
  pt::ptree options;
  LinkFunction* lf;
public:

  virtual double compute_log_p(double z, double param) {
    if (z == 1)
      return -lf->f(-param);
    else
      return -lf->f(param);
  }

  BernoulliLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    lf = get_link_function(options.get<string>("layer.lf", "softmax"));
  }
};

class BernoulliPriorLayer : public DEFPriorLayer {

protected:
  pt::ptree options;
  double prob;
  LinkFunction* lf;
public:

  BernoulliPriorLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    prob = options.get<double>("layer.prob");
  }

  virtual double compute_log_p(double z) {
    if (z == 1)
      return log(prob);
    else
      return log(1-prob);
  }

};

class BernoulliFactorizedLayer : public InferenceFactorizedLayer {
protected:
  arma::uword layer_size;
  Serializable<arma::mat> w;
  LinkFunction* lf;
public:

  virtual double compute_log_q(double z, arma::uword i, arma::uword j) {
    auto log_q = z == 1 ? -softmax(-w(i,j)) : -softmax(w(i,j));
    LOG_IF(fatal, !isfinite(log_q))
      << "w=" << w(i,j)
      << " z=" << z << " log_q=" << log_q;
    assert(isfinite(log_q));
    return log_q;
  }  

  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) {
    auto p = sigmoid(w(i,j));
    auto z = gsl_ran_bernoulli(rng, p);
    return z;
  }

  virtual double mean(arma::uword i, arma::uword j) {
    auto p = sigmoid(w(i,j));
    return p;
  }

  virtual void copy_params(InferenceFactorizedLayer* other) {
    BernoulliFactorizedLayer* other_bfl = dynamic_cast<BernoulliFactorizedLayer*>(other);
    if (other_bfl == NULL)
      throw runtime_error("can't cast to BernoulliFactorizedLayer");
    w = other_bfl->w;
  }

  virtual void truncate(const ExampleIds& example_ids) {
    // no op
  }

  virtual void truncate() {
    truncate(all_examples);
  }

  BernoulliFactorizedLayer() {}

  BernoulliFactorizedLayer(const pt::ptree& options,
                           const DEFInitializer& initializer)
    : InferenceFactorizedLayer(options) {    
    init(false);
 
    gsl_rng* rng = initializer.rng;   
    auto w_init = options.get<double>("layer.w_init");
    for(auto& v : w) {
      v = gsl_ran_gaussian(rng, 1) * w_init;
    }
  }

  void init(bool deserialize) {
    layer_size = options.get<int>("layer.size");
    lf = get_link_function(options.get<string>("lf"));
    w.set_size(layer_size, n_examples);

    ScoreFunction score = [=](double z, arma::uword i, arma::uword j) {
      if (z == 1)
        return 1-sigmoid(w(i,j));
      else
        return -sigmoid(w(i,j));
    };
    register_param(&w, score, deserialize);
  }
  
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & w;
    ar & boost::serialization::base_object<const InferenceFactorizedLayer>(*this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & w;
    ar & boost::serialization::base_object<InferenceFactorizedLayer>(*this);
    init(true);
  }
};

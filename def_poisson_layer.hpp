#pragma once

#include <cassert>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include "utils.hpp"
#include "def_layer.hpp"
#include "link_function.hpp"
#include "serialization.hpp"

// p(z_i | z_{i+1}), E[z_i] = W*z_{i+1}
class PoissonLayer : public DEFLayer {
private:
  pt::ptree options;
  double poisson_rate_intercept;
  LinkFunction *lf;

public:
  virtual double compute_log_p(double z, double param) {
    param = lf->f(param);
    param += poisson_rate_intercept;
    return -param + z * log(param) - gsl_sf_lngamma(z + 1);
  }

  PoissonLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    poisson_rate_intercept = options.get<double>("global.poisson_rate_intercept");
    lf = get_link_function(options.get<string>("layer.lf"));
  }
};

class PoissonPriorLayer : public DEFPriorLayer {

protected:
  pt::ptree options;
  double rate;
public:

  PoissonPriorLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    rate = options.get<double>("layer.rate");
  }

  virtual double compute_log_p(double z) {
    return -rate + z * log(rate) - gsl_sf_lngamma(z + 1);
  }
};

class PoissonFactorizedLayer : public InferenceFactorizedLayer {
protected:
  arma::uword layer_size;
  Serializable<arma::mat> wrate;
  LinkFunction* lf;
public:

  virtual double compute_log_q(double z, arma::uword i, arma::uword j) {
    auto rate = lf->f(wrate(i,j));
    auto log_q = -rate + z * log(rate) - gsl_sf_lngamma(z + 1);
    LOG_IF(fatal, !isfinite(log_q))
           << "rate=" << rate
           << " z=" << z << " log_q=" << log_q;
    assert(isfinite(log_q));
    return log_q;
  }  

  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) {
    auto rate = lf->f(wrate(i,j));
    auto z = gsl_ran_poisson(rng, rate);
    LOG_IF(fatal, (!isfinite(z)))
           << "rate=" << rate
           << " z=" << z;
    return z;
  }

  virtual double mean(arma::uword i, arma::uword j) {
    return lf->f(wrate(i,j));
  }

  virtual void copy_params(InferenceFactorizedLayer* other) {
    PoissonFactorizedLayer* other_pfl = dynamic_cast<PoissonFactorizedLayer*>(other);
    if (other_pfl == NULL)
      throw runtime_error("can't cast to GammaFactorizedLayer");
    wrate = other_pfl->wrate;
  }

  virtual void truncate(const ExampleIds& example_ids) {
    auto min_rate0 = lf->f_inv(options.get<double>("global.min_poisson_rate"));
    for(auto j : example_ids) {
      wrate.col(j).transform([=](double v) { return max(v, min_rate0); });
    }
  }

  virtual void truncate() {
    truncate(all_examples);
  }

  PoissonFactorizedLayer() {}

  PoissonFactorizedLayer(const pt::ptree& options,
			 const DEFInitializer& initializer)
    : InferenceFactorizedLayer(options) {    
    init(false);
 
    gsl_rng* rng = initializer.rng;   
    auto wrate_init = options.get<double>("layer.wrate_init");
    for(auto& v : wrate) {
      v = exp(gsl_ran_gaussian(rng, 1)) * wrate_init;
    }
  }

  void init(bool deserialize) {
    LOG(debug) << "global.min_poisson_rate="
               << options.get<double>("global.min_poisson_rate");
    layer_size = options.get<int>("layer.size");
    lf = get_link_function(options.get<string>("lf"));
    wrate.set_size(layer_size, n_examples);

    ScoreFunction score_rate = [=](double z, arma::uword i, arma::uword j) {
      auto rate0 = wrate(i,j);
      auto rate = lf->f(rate0);
      return -1 + z / rate;
    };
    register_param(&wrate, score_rate, deserialize);
  }
  
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & wrate;
    ar & boost::serialization::base_object<const InferenceFactorizedLayer>(*this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & wrate;
    ar & boost::serialization::base_object<InferenceFactorizedLayer>(*this);
    init(true);
  }
};


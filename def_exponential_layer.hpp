#pragma once

#include <cassert>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include "utils.hpp"
#include "def_layer.hpp"
#include "link_function.hpp"
#include "serialization.hpp"

// the prior will always be symmetric and 0-centered
class ExponentialPriorLayer : public DEFPriorLayer {

protected:
  pt::ptree options;
  double scale;
  double log_scale;
  LinkFunction *lf;

public:
  ExponentialPriorLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    scale = options.get<double>("layer.exp_scale");
    log_scale = log(scale);
    lf = get_link_function(options.get<string>("layer.q_lf"));
  }

  virtual double compute_log_p(double z) {
    // log(1.0/scale * exp(-x/scale)) = - log_scale - x / scale
    z = lf->f(z);
    return -log_scale - z / scale;
  }
};

class ExponentialFactorizedLayer : public InferenceFactorizedLayer {
protected:
  arma::uword layer_size;
  Serializable<arma::mat> w_scale;
  double exp_scale_intercept;
  LinkFunction *lf;

public:
  virtual double compute_log_q(double z, arma::uword i, arma::uword j) {
    auto scale = lf->f(w_scale(i,j));
    auto log_q = -log(scale) - z/scale;
    LOG_IF(fatal, !isfinite(log_q))
           << "scale=" << scale
           << " z=" << z << " log_q=" << log_q;
    assert(isfinite(log_q));
    return log_q;
  }

  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) {
    // mu always uses the identity link function
    auto scale = lf->f(w_scale(i,j));
    auto z = gsl_ran_exponential(rng, scale);
    return z;
  }

  virtual double mean(arma::uword i, arma::uword j) {
    return lf->f(w_scale(i,j));
  }

  virtual void copy_params(InferenceFactorizedLayer* other) {
    ExponentialFactorizedLayer* other_gfl = dynamic_cast<ExponentialFactorizedLayer*>(other);
    if (other_gfl == NULL)
      throw runtime_error("can't cast to ExponentialFactorizedLayer");
    w_scale = other_gfl->w_scale;
  }

  virtual void truncate(const ExampleIds& example_ids) {
    auto min_scale0 = lf->f_inv(options.get<double>("global.min_weight_exp_scale"));

    for(auto j : example_ids) {
      w_scale.col(j).transform([=](double v) { return max(v, min_scale0); });
    }
  }

  virtual void truncate() {
    truncate(all_examples);
  }

  ExponentialFactorizedLayer() {}

  ExponentialFactorizedLayer(const pt::ptree& options,
                       const DEFInitializer& initializer)
    : InferenceFactorizedLayer(options) {
    init(false);

    gsl_rng* rng = initializer.rng;
    auto w_scale_init = options.get<double>("global.weight_exp_scale_init");
    for(auto& v : w_scale) {
      v = gsl_ran_gaussian(rng, 1) * w_scale_init;
    }
  }

  void init(bool deserialize) {
    layer_size = options.get<int>("layer.size");
    LOG(debug) << "def exponential factorized layer init, size=" << layer_size;
    lf = get_link_function(options.get<string>("lf"));
    w_scale.set_size(layer_size, n_examples);

    ScoreFunction score_scale = [=](double z, arma::uword i, arma::uword j) {
      auto scale0 = w_scale(i,j);
      auto scale = lf->f(scale0);
      return lf->g(scale0) * (-1.0/scale + z / (scale * scale));
    };
    register_param(&w_scale, score_scale, deserialize);
  }

  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & w_scale;
    ar & boost::serialization::base_object<const InferenceFactorizedLayer>(*this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & w_scale;
    ar & boost::serialization::base_object<InferenceFactorizedLayer>(*this);
    init(true);
  }
};

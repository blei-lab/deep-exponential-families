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
class GammaLayer : public DEFLayer {

private:
  pt::ptree options;
  double shape;
  LinkFunction* lf;
  double min_gamma_scale;
public:

  virtual double compute_log_p(double z, double param) {
    double scale = lf->f(param) / shape + min_gamma_scale;
    return - gsl_sf_lngamma(shape) - shape*log(scale) + (shape-1)*log(z) - z/scale;
  }
  
  GammaLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    shape = options.get<double>("layer.shape");
    min_gamma_scale = options.get<double>("layer.min_gamma_scale");
    lf = get_link_function(options.get<string>("layer.lf"));
  }
};

class GammaPriorLayer : public DEFPriorLayer {

protected:
  pt::ptree options;
  double shape, scale;
  LinkFunction* lf;
public:

  GammaPriorLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    shape = options.get<double>("layer.shape");
    scale = options.get<double>("layer.scale");
  }

  virtual double compute_log_p(double z) {
    return - gsl_sf_lngamma(shape) - shape*log(scale) + (shape-1)*log(z) - z/scale;
  }
};

class GammaFactorizedLayer : public InferenceFactorizedLayer {
protected:
  arma::uword layer_size;
  Serializable<arma::mat> wshape, wscale;
  LinkFunction* lf;
  double min_gamma_sample;
public:

  virtual double compute_log_q(double z, arma::uword i, arma::uword j) {
    auto shape = lf->f(wshape(i,j));
    auto scale = lf->f(wscale(i,j));
    auto log_q = - gsl_sf_lngamma(shape) - shape*log(scale) + (shape-1)*log(z) - z/scale;
    LOG_IF(fatal, !isfinite(log_q))
           << "shape=" << shape << " scale=" << scale
           << " z=" << z << " log_q=" << log_q;
    assert(isfinite(log_q));
    return log_q;
  }  

  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) {
    auto shape = lf->f(wshape(i,j));
    auto scale = lf->f(wscale(i,j));
    auto z = gsl_ran_gamma(rng, shape, scale);
    //LOG_IF(fatal, (z < 1e-320) || (!isfinite(z)))
    //       << "shape=" << shape
    //      << " scale=" << scale
    //       << " z=" << z;
    z = max(z, min_gamma_sample);
    assert(z >= 1e-300);
    return z;
  }

  virtual double mean(arma::uword i, arma::uword j) {
    auto shape = lf->f(wshape(i,j));
    auto scale = lf->f(wscale(i,j));
    return shape*scale;
  }

  virtual void copy_params(InferenceFactorizedLayer* other) {
    GammaFactorizedLayer* other_gfl = dynamic_cast<GammaFactorizedLayer*>(other);
    if (other_gfl == NULL)
      throw runtime_error("can't cast to GammaFactorizedLayer");
    wshape = other_gfl->wshape;
    wscale = other_gfl->wscale;
  }

  virtual void truncate(const ExampleIds& example_ids) {
    auto min_shape0 = lf->f_inv(options.get<double>("global.min_gamma_shape"));
    auto min_scale0 = lf->f_inv(options.get<double>("global.min_gamma_scale"));

    for(auto j : example_ids) {
      wshape.col(j).transform([=](double v) { return max(v, min_shape0); });
      wscale.col(j).transform([=](double v) { return max(v, min_scale0); });
    }
  }

  virtual void truncate() {
    truncate(all_examples);
  }

  GammaFactorizedLayer() {}

  GammaFactorizedLayer(const pt::ptree& options,
                       const DEFInitializer& initializer)
    : InferenceFactorizedLayer(options) {    
    init(false);
 
    gsl_rng* rng = initializer.rng;   
    auto wshape_init = options.get<double>("layer.wshape_init");
    for(auto& v : wshape) {
      v = exp(gsl_ran_gaussian(rng, 1)) * wshape_init;
    }
    auto wscale_init = options.get<double>("layer.wscale_init");
    for(auto& v : wscale) {
      v = exp(gsl_ran_gaussian(rng, 1)) * wscale_init;      
    }

  }

  void init(bool deserialize) {
    LOG(debug) << "global.min_gamma_shape="
               << options.get<double>("global.min_gamma_shape");
    layer_size = options.get<int>("layer.size");
    lf = get_link_function(options.get<string>("lf"));
    min_gamma_sample = options.get<double>("global.min_gamma_sample");
    wshape.set_size(layer_size, n_examples);
    wscale.set_size(layer_size, n_examples);

    ScoreFunction score_shape = [=](double z, arma::uword i, arma::uword j) {
      auto shape0 = wshape(i,j);
      auto shape = lf->f(shape0);
      auto scale = lf->f(wscale(i,j));
      return lf->g(shape0) * (- gsl_sf_psi(shape) - log(scale) + log(z));
    };
    register_param(&wshape, score_shape, deserialize);

    ScoreFunction score_scale = [=](double z, arma::uword i, arma::uword j) {
      auto shape = lf->f(wshape(i,j));
      auto scale0 = wscale(i,j);
      auto scale = lf->f(scale0);
      return lf->g(scale0) * (- shape/scale + z/scale/scale);
    };
    register_param(&wscale, score_scale, deserialize);
  }
  
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & wshape;
    ar & wscale;
    ar & boost::serialization::base_object<const InferenceFactorizedLayer>(*this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & wshape;
    ar & wscale;
    ar & boost::serialization::base_object<InferenceFactorizedLayer>(*this);
    init(true);
  }
};

#pragma once

#include <cassert>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include "utils.hpp"
#include "def_layer.hpp"
#include "link_function.hpp"
#include "serialization.hpp"

// the prior will always be 0-centered
class GaussianPriorLayer : public DEFPriorLayer {

protected:
  pt::ptree options;
  double mu;
  double sigma;
  double log_sqrt_2pi;
  double sigma2;

public:
  GaussianPriorLayer(const pt::ptree& options, const DEFInitializer& initializer)
    : options( options ) {
    mu = options.get<double>("layer.mu", 0.0);
    if (mu != 0.0) {
      LOG(debug) << "gaussian prior layer mu=" << mu;
      assert(mu == options.get<double>("layer.w_mu_init_offset"));
    }
    sigma = options.get<double>("layer.sigma");
    log_sqrt_2pi = log(sqrt(2*M_PI));
    sigma2 = sigma * sigma;
  }

  virtual double compute_log_p(double z) {
    z -= mu;
    return - log_sqrt_2pi - log(sigma) - z*z / (2*sigma2);
  }
};

class GaussianFactorizedLayer : public InferenceFactorizedLayer {
protected:
  arma::uword layer_size;
  Serializable<arma::mat> w_mu, w_sigma;
  LinkFunction* lf;
public:

  virtual double compute_log_q(double z, arma::uword i, arma::uword j) {
    // mu always uses the identity link function
    auto mu = w_mu(i,j);
    auto sigma = lf->f(w_sigma(i,j));
    auto log_q = - log(2*M_PI)*0.5 - log(sigma) - ((z-mu)*(z-mu)) / (2*sigma*sigma);
    LOG_IF(fatal, !isfinite(log_q))
           << "mu=" << mu << " sigma=" << sigma
           << " z=" << z << " log_q=" << log_q;
    assert(isfinite(log_q));
    return log_q;
  }

  virtual double sample(gsl_rng* rng, arma::uword i, arma::uword j) {
    // mu always uses the identity link function
    auto mu = w_mu(i,j);
    auto sigma = lf->f(w_sigma(i,j));
    auto z = gsl_ran_gaussian(rng, sigma) + mu;
    return z;
  }

  virtual double mean(arma::uword i, arma::uword j) {
    return w_mu(i,j);
  }

  virtual void copy_params(InferenceFactorizedLayer* other) {
    GaussianFactorizedLayer* other_gfl = dynamic_cast<GaussianFactorizedLayer*>(other);
    if (other_gfl == NULL)
      throw runtime_error("can't cast to GaussianFactorizedLayer");
    w_mu = other_gfl->w_mu;
    w_sigma = other_gfl->w_sigma;
  }

  virtual void truncate(const ExampleIds& example_ids) {
    // fixed variance
    if (options.get("global.fixed_gaussian_sigma", 0.0)) {
      auto fixed_sigma = options.get<double>("global.fixed_gaussian_sigma");
      auto fixed_sigma0 = lf->f_inv(fixed_sigma);
      for(auto j : example_ids) {
        w_sigma.col(j).transform([=](double s) { return fixed_sigma0; });
      }
    }
    else { // no fixed variance
      double min_sigma = options.get("global.min_gaussian_sigma", 0.0);
      // no truncation
      if (min_sigma == 0)
        return;

      auto min_sigma0 = lf->f_inv(min_sigma);

      for(auto j : example_ids) {
        w_sigma.col(j).transform([=](double v) { return max(v, min_sigma0); });
      }
    }
  }

  virtual void truncate() {
    truncate(all_examples);
  }

  GaussianFactorizedLayer() {}

  GaussianFactorizedLayer(const pt::ptree& options,
                       const DEFInitializer& initializer)
    : InferenceFactorizedLayer(options) {
    init(false);

    gsl_rng* rng = initializer.rng;
    auto w_mu_init = options.get<double>("layer.w_mu_init");
    auto w_mu_init_offset = options.get<double>("layer.w_mu_init_offset", 0.0);
    if (w_mu_init_offset != 0) {
      LOG(debug) << "guassian factorized layer w_mu_init_offset=" << w_mu_init_offset;
      assert(w_mu_init_offset == options.get<double>("layer.mu"));
    }
    for(auto& v : w_mu) {
      // use gaussian to initilize mu
      v = gsl_ran_gaussian(rng, 1) * w_mu_init + w_mu_init_offset;
    }
    auto w_sigma_init = options.get<double>("layer.w_sigma_init");
    for(auto& v : w_sigma) {
      // use gaussian to intilize sigma before the link_function
      v = gsl_ran_gaussian(rng, 1) * w_sigma_init;
    }

    auto min_sigma = options.get("global.min_gaussian_sigma", 0.0);
    if (min_sigma > 0)
      LOG(debug) << "global.min_gaussian_sigma=" << min_sigma;

    auto fixed_sigma = options.get("global.fixed_gaussian_sigma", 0.0);
    if (fixed_sigma > 0) {
      LOG(debug) << "global.fixed_gaussian_sigma=" << fixed_sigma;
    }
  }



  void init(bool deserialize) {
    layer_size = options.get<int>("layer.size");
    lf = get_link_function(options.get<string>("lf"));
    w_mu.set_size(layer_size, n_examples);
    w_sigma.set_size(layer_size, n_examples);

    ScoreFunction score_mu = [=](double z, arma::uword i, arma::uword j) {
      auto mu = w_mu(i,j);
      auto sigma = lf->f(w_sigma(i,j));
      return (z-mu) / (sigma*sigma);
    };
    register_param(&w_mu, score_mu, deserialize);

    ScoreFunction score_sigma = [=](double z, arma::uword i, arma::uword j) {
      auto mu = w_mu(i,j);
      auto sigma0 = w_sigma(i,j);
      auto sigma = lf->f(sigma0);
      return lf->g(sigma0) * (-1.0/sigma + (z-mu)*(z-mu) / (sigma*sigma*sigma));
    };
    register_param(&w_sigma, score_sigma, deserialize);
  }

  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & w_mu;
    ar & w_sigma;
    ar & boost::serialization::base_object<const InferenceFactorizedLayer>(*this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & w_mu;
    ar & w_sigma;
    ar & boost::serialization::base_object<InferenceFactorizedLayer>(*this);
    init(true);
  }
};

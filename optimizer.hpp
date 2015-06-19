#pragma once

#include "utils.hpp"
#include "serialization.hpp"
#include <signal.h>

class Optimizer {
private:
  ExampleIds all_examples;
  Serializable<arma::mat>* w;
  Serializable<arma::mat> G, V, Tau;
  string algo;
  double rho;
  double tau;
public:
  Optimizer(const pt::ptree& options, Serializable<arma::mat>* w) : 
    w( w ), 
    G(w->n_rows, w->n_cols, arma::fill::zeros),
    V(w->n_rows, w->n_cols, arma::fill::zeros),
    Tau(w->n_rows, w->n_cols, arma::fill::ones),
    algo(options.get<string>("algo")),
    rho(options.get<double>("rho")),
    tau(options.get<double>("tau"))
  {
    setup();
  }

  void setup() {
    all_examples.clear();
    for (arma::uword i=0; i<w->n_cols; ++i)
      all_examples.push_back(i);
  }

  void update(const arma::mat& g) {
    update(g, all_examples);
  }

  void update(const arma::mat& g, const ExampleIds& example_ids) {
    if(algo == "ada")
      ada_ascent(g, example_ids);
    else if (algo == "rmsprop")
      rmsprop_ascent(g, example_ids);
    else if (algo == "vsgd")
      vsgd_ascent(g, example_ids);
    else
      throw runtime_error("unknown optimization algorithm");
  }

  void ada_ascent(const arma::mat& g, const ExampleIds& example_ids);

  void rmsprop_ascent(const arma::mat& g, const ExampleIds& example_ids);

  void vsgd_ascent(const arma::mat& g, const ExampleIds& example_ids);


  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & algo;
    ar & rho;
    ar & tau;
    ar & G;
    ar & V;
    ar & Tau;
    ar & w;
    setup();
  }
  Optimizer() : w(NULL), algo("") {}
};


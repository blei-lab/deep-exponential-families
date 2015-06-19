#include "optimizer.hpp"

void Optimizer::ada_ascent(const arma::mat& g, const ExampleIds& example_ids) {
  arma::uword j0 = 0;
  for (auto j : example_ids) {
    G.col(j) += g.col(j0) % g.col(j0);
    ++j0;
  }
    
  j0 = 0;
  for (auto j : example_ids) {
    for (arma::uword i=0; i<G.n_rows; ++i) {
      if (G(i,j) > 0)
        (*w)(i,j) += rho / sqrt(G(i,j)) * g(i,j0);
    }
    ++j0;
  }
}

void Optimizer::rmsprop_ascent(const arma::mat& g, const ExampleIds& example_ids) {
  auto inv_tau = 1.0/tau;
  arma::uword j0 = 0;
  for (auto j : example_ids) {
    for(arma::uword i=0; i<G.n_rows; ++i) {
      auto G0 = G(i,j);
      if (G0 == 0)
        G(i, j) = g(i, j0) * g(i, j0);
      else
        G(i, j) = (1.0-inv_tau) * G0 + inv_tau * g(i, j0) * g(i, j0);

      if (G(i,j) > 0)
        (*w)(i,j) += rho / sqrt(G(i,j)) * g(i,j0);        
    }
    ++j0;
  }    
}

void Optimizer::vsgd_ascent(const arma::mat& g, const ExampleIds& example_ids) {
  arma::uword j0 = 0;
  for (auto j : example_ids) {
    for(arma::uword i=0; i<G.n_rows; ++i) {
      auto G0 = G(i,j);
      auto V0 = V(i,j);
      auto inv_tau = 1.0/Tau(i,j);
      // update G & V
      if (G0 == 0) {
        G(i, j) = g(i, j0) * g(i, j0);
        V(i, j) = g(i, j0);
      }
      else {
        G(i, j) = (1.0-inv_tau) * G0 + inv_tau * g(i, j0) * g(i, j0);
        V(i, j) = (1.0-inv_tau) * V0 + inv_tau * g(i, j0);
      }

      // update w and Tau
      if (G(i,j) > 0) {
        (*w)(i,j) += rho * abs(V(i,j)) / G(i,j) * g(i,j0);
        Tau(i,j) = max((1.0-V(i,j)*V(i,j)/G(i,j)) * Tau(i,j) + 1.0, 3.0);
      }
    }
    ++j0;
  }    
}

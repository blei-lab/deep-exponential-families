#include "def_bernoulli_layer.hpp"
#include "layer_factory.hpp"

BOOST_CLASS_EXPORT_GUID(BernoulliFactorizedLayer, "BernoulliFactorizedLayer")
REGISTER_P_Z_LAYER("bernoulli", BernoulliLayer)
REGISTER_PRIOR_LAYER("bernoulli", BernoulliPriorLayer)
REGISTER_Q_LAYER("bernoulli", BernoulliFactorizedLayer)

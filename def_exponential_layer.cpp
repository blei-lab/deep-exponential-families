#include "def_exponential_layer.hpp"
#include "layer_factory.hpp"

BOOST_CLASS_EXPORT_GUID(ExponentialFactorizedLayer, "ExponentialFactorizedLayer")
// REGISTER_P_Z_LAYER("exp", ExponentialLayer)
REGISTER_PRIOR_LAYER("exp", ExponentialPriorLayer)
REGISTER_Q_LAYER("exp", ExponentialFactorizedLayer)

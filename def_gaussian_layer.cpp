#include "def_gaussian_layer.hpp"
#include "layer_factory.hpp"

BOOST_CLASS_EXPORT_GUID(GaussianFactorizedLayer, "GaussianFactorizedLayer")
// REGISTER_P_Z_LAYER("gaussian", GaussianLayer)
REGISTER_PRIOR_LAYER("gaussian", GaussianPriorLayer)
REGISTER_Q_LAYER("gaussian", GaussianFactorizedLayer)

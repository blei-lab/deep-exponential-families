#include "def_gamma_layer.hpp"
#include "layer_factory.hpp"

BOOST_CLASS_EXPORT_GUID(GammaFactorizedLayer, "GammaFactorizedLayer")
REGISTER_P_Z_LAYER("gamma", GammaLayer)
REGISTER_PRIOR_LAYER("gamma", GammaPriorLayer)
REGISTER_Q_LAYER("gamma", GammaFactorizedLayer)

#include "def_poisson_layer.hpp"
#include "layer_factory.hpp"

BOOST_CLASS_EXPORT_GUID(PoissonFactorizedLayer, "PoissonFactorizedLayer")
REGISTER_P_Z_LAYER("poisson", PoissonLayer)
REGISTER_PRIOR_LAYER("poisson", PoissonPriorLayer)
REGISTER_Q_LAYER("poisson", PoissonFactorizedLayer)

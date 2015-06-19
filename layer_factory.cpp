#include "layer_factory.hpp"

map<string, DEFPriorLayerFactory>* def_prior_map;
map<string, DEFLayerFactory>* def_p_z_map;
map<string, InferenceFactorizedLayerFactory>* def_factorized_inference_map;
map<string, DEFYLayerFactory>* def_p_y_map;

static int nifty_counter; 

FactoryMapInitializer::FactoryMapInitializer() {
  if (0 == nifty_counter++) {
    def_p_y_map = new map<string, DEFYLayerFactory>();
    def_p_z_map = new map<string, DEFLayerFactory>();
    def_prior_map = new map<string, DEFPriorLayerFactory>();
    def_factorized_inference_map = new map<string, InferenceFactorizedLayerFactory>();
  }
}

FactoryMapInitializer::~FactoryMapInitializer() {
  if (0 == --nifty_counter) {
    delete def_p_y_map;
    delete def_p_z_map;
    delete def_prior_map;
    delete def_factorized_inference_map;
  }
}

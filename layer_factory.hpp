#pragma once

#include "utils.hpp"
#include "def_layer.hpp"
#include "def_y_layer.hpp"

typedef function<shared_ptr<DEFLayer>(const pt::ptree&, const DEFInitializer&) > DEFLayerFactory;
typedef function<shared_ptr<DEFPriorLayer>(const pt::ptree&, const DEFInitializer&) > DEFPriorLayerFactory;
typedef function<shared_ptr<InferenceFactorizedLayer>(const pt::ptree&,
                                                      const DEFInitializer&) > InferenceFactorizedLayerFactory;
typedef function<shared_ptr<DEFYLayer>(const pt::ptree&,
                                       const DEFInitializer&) > DEFYLayerFactory;


extern map<string, DEFLayerFactory>* def_p_z_map;
extern map<string, DEFPriorLayerFactory>* def_prior_map;
extern map<string, InferenceFactorizedLayerFactory>* def_factorized_inference_map;
extern map<string, DEFYLayerFactory>* def_p_y_map;


// Nifty Counter to ensure initialization of the map
static struct FactoryMapInitializer {
  FactoryMapInitializer();
  ~FactoryMapInitializer();
} initialzer;

// construct layers
#define GET_P_Y_LAYER(layer_name, options, initializer) \
  ((*def_p_y_map)[layer_name](options, initializer))
#define GET_P_Z_LAYER(layer_name, options, initializer) \
  ((*def_p_z_map)[layer_name](options, initializer))
#define GET_PRIOR_LAYER(layer_name, options, initializer)       \
  ((*def_prior_map)[layer_name](options, initializer))
#define GET_Q_LAYER(layer_name, options, initializer)           \
  ((*def_factorized_inference_map)[layer_name](options, initializer))


// register layers
#define REGISTER_P_Y_LAYER(layer_name, type)                            \
  struct factory_##type {                                               \
    static shared_ptr<DEFYLayer>                                         \
    create_##type(const pt::ptree& m, const DEFInitializer& i) {        \
      return shared_ptr<DEFYLayer>( new type(m, i) );                   \
    }                                                                   \
    factory_##type() {                                                  \
      (*def_p_y_map)[layer_name] = create_##type;                       \
    }                                                                   \
  } factory_##type##_x;                                                   


#define REGISTER_P_Z_LAYER(layer_name, type)                            \
  struct factory_##type {                                               \
    static shared_ptr<DEFLayer>                                         \
    create_##type(const pt::ptree& m, const DEFInitializer& i) {        \
      return shared_ptr<DEFLayer>( new type(m, i) );                    \
    }                                                                   \
    factory_##type() {                                                  \
      (*def_p_z_map)[layer_name] = create_##type;                       \
    }                                                                   \
  } factory_##type##_x;                                                   

#define REGISTER_PRIOR_LAYER(layer_name, type)                          \
  struct factory_##type {                                               \
    static shared_ptr<DEFPriorLayer>                                    \
    create_##type(const pt::ptree& m, const DEFInitializer& i) {        \
      return shared_ptr<DEFPriorLayer>( new type(m, i) );               \
    }                                                                   \
    factory_##type() {                                                  \
      (*def_prior_map)[layer_name] = create_##type;                     \
    }                                                                   \
  } factory_##type##_x;

#define REGISTER_Q_LAYER(layer_name, type)                              \
  struct factory_q_##type {                                             \
    static shared_ptr<InferenceFactorizedLayer>                         \
    create_##type(const pt::ptree& m, const DEFInitializer& i) {        \
      return shared_ptr<InferenceFactorizedLayer>( new type(m, i) );    \
    }                                                                   \
    factory_q_##type() {                                                \
      (*def_factorized_inference_map)[layer_name] = create_##type;      \
    }                                                                   \
  } factory_q_##type##_x;

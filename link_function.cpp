#include "link_function.hpp"

std::unordered_map<string, shared_ptr<LinkFunction> > lf_map;
LinkFunction*  get_link_function(const string& lf_name) {
  if (lf_map.count("shifted_softmax") == 0)
    lf_map["shifted_softmax"] = shared_ptr<LinkFunction>( new ShiftedSoftMax() );
  if (lf_map.count("softmax") == 0)
    lf_map["softmax"] = shared_ptr<LinkFunction>( new SoftMax() );
  if (lf_map.count("id") == 0)
    lf_map["id"] = shared_ptr<LinkFunction>( new IdentityLink() );
  assert(lf_map.count(lf_name) > 0);
  return &*lf_map.at(lf_name);
}

void init_shifted_softmax(const pt::ptree& ptree) {
  double shift = ptree.get("global.softmax_shift", 10.0f);
  ShiftedSoftMax* ssm = (ShiftedSoftMax*) get_link_function("shifted_softmax");
  ssm->shift = shift;
  LOG(debug) << "softmax shift=" << shift;
}

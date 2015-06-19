#ifndef RANDOM
#define RANDOM

#include <boost/noncopyable.hpp>
#include <gsl/gsl_rng.h>

// Not Copy Safe
struct GSLRandom : boost::noncopyable
{
public:
  inline GSLRandom() {
    rng = gsl_rng_alloc(gsl_rng_taus);
  }
  ~GSLRandom() {
    gsl_rng_free(rng);
  }
  gsl_rng* rng;

  // Serialization
  // WARNING:
  // Reaches into GSL internals. It assumes that the size of the state never changes
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  template<typename Archive>
  void save(Archive& ar, unsigned int) const {
    size_t size = rng->type->size;
    const char* mem = static_cast<char*>(rng->state);

    ar & size;
    for (size_t i = 0; i < size; ++i) {
      ar & mem[i];
    }
  }

  template<typename Archive>
  void load(Archive& ar, unsigned int) {
    size_t size;
    ar & size;

    assert(rng->type->size == size);
    char* mem = new char[size];
    for (size_t i = 0; i < size; ++i) {
      ar & mem[i];
    }

    memcpy(rng->state, mem, size);
    delete[] mem;
  }
};

#endif

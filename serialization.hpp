#ifndef SERIALIZATION
#define SERIALIZATION

// Rajesh Ranganth (2014)
#define ARMA_64BIT_WORD
#include <armadillo>

// Simple access headers
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/access.hpp>

// A simple, portable text archive
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// A binary archive
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// Utilities
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/export.hpp>

// STL
#include <boost/serialization/vector.hpp>

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>         //allows the use of filters like gzip that are automatically applied when reading from or writing to a file                                                                                                                  
#include <boost/iostreams/filter/gzip.hpp>              //allows gzip (de)compression  

// Vector
template<typename Archive>
void serialize_helper(Archive& ar, const arma::vec& v) {
  ar & v.n_elem;
  for (arma::uword i = 0; i < v.n_elem; ++i)
    ar & v(i);
}

template<typename Archive>
void deserialize_helper(Archive& ar, arma::vec* v) {
  arma::uword n_elem;
  ar & n_elem;
  v->resize(n_elem);
  for (arma::uword i = 0; i < v->n_elem; ++i)
    ar & (*v)(i);
}

// Matrix
template<typename Archive>
void serialize_helper(Archive& ar, const arma::mat& m) {
  ar & m.n_rows;
  ar & m.n_cols;

  for (arma::uword r = 0; r < m.n_rows; ++r)
    for (arma::uword c = 0; c < m.n_cols; ++c)
      ar & m(r, c);
}

template<typename Archive>
void deserialize_helper(Archive& ar, arma::mat* m) {
  arma::uword n_rows, n_cols;
  ar & n_rows;
  ar & n_cols;
  m->set_size(n_rows, n_cols);

  for (arma::uword r = 0; r < m->n_rows; ++r)
    for (arma::uword c = 0; c < m->n_cols; ++c)
      ar & (*m)(r, c);
}

// Work around for arma serialization to avoid having to define save and load everywhere
template<typename T>
class Serializable: public T {
 public:
  inline Serializable() {}

  template<typename... Args>
  inline Serializable(Args&&... args) : T(args...) {}

  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  // Sometimes you need explicit casts for template libraries with operators...
  inline T& v() { return *static_cast<T*>(this); }
  inline const T& v() const { return *static_cast<const T*>(this); }

  // Serialization                                                                                                                                                                                                                                                                   
  template<class Archive>
  inline void save(Archive& ar, const unsigned int) const {
    serialize_helper(ar, *static_cast<const T*>(this));
  }

  template<class Archive>
  inline void load(Archive& ar, const unsigned int) {
    deserialize_helper(ar, static_cast<T*>(this));
  }
};

// Writes (Erase and Writes) a file containing the object
template<typename Archive, typename Object>
inline void serialize(const std::string& filename, const Object& o) {
  std::ofstream ofs(filename.c_str());
  Archive oa(ofs);
  oa << o;
}

template<typename Archive, typename Object>
inline void deserialize(const std::string& filename, Object* o) {
  std::ifstream ifs(filename.c_str());
  Archive ia(ifs);
  ia >> *o;
}

template<typename Archive, typename Object>
inline void serialize_gzip(const std::string& filename, const Object& o) {
  std::ofstream ofs(filename.c_str());
  boost::iostreams::filtering_stream<boost::iostreams::output> f;
  f.push(boost::iostreams::gzip_compressor());
  f.push(ofs);
  Archive oa(f);
  oa << o;
}

template<typename Archive, typename Object>
inline void deserialize_gzip(const std::string& filename, Object* o) {
  std::ifstream ifs(filename.c_str());
  boost::iostreams::filtering_stream<boost::iostreams::input> f;
  f.push(boost::iostreams::gzip_decompressor());
  f.push(ifs);
  
  Archive ia(f);
  ia >> *o;
}

typedef boost::archive::text_iarchive state_iarchive;
typedef boost::archive::text_oarchive state_oarchive;

#endif

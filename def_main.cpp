#include "utils.hpp"
#include "layer_factory.hpp"
#include "def_model.hpp"
#include "def_2_model.hpp"
#include "def_gamma_layer.hpp"
#include <chrono>

// Implementations: TODO this should be moved elsewhere
#include <boost/archive/impl/basic_text_oarchive.ipp>
#include <boost/archive/impl/text_oarchive_impl.ipp>
#include <boost/archive/impl/archive_serializer_map.ipp>

#include <boost/archive/impl/basic_text_iarchive.ipp>
#include <boost/archive/impl/text_iarchive_impl.ipp>
#include <boost/archive/impl/basic_text_iprimitive.ipp>

// parse command line and config file and store the result in vm and ptree
pt::ptree parse_options(int argc, char* argv[]) {
  po::options_description desc("Deep Exponential Family");
  desc.add_options()
    ("help", "produce help message")
    ("v", po::value<int>()->default_value(1), "log level, default:1 (warning)")
    ("folder", po::value<string>(), "folder to put the log, model, etc.")
    ("time", po::value<bool>()->default_value(true), "append time to experiment name")
    ("deserialization_test", po::value<bool>()->default_value(false), "Test deserialization by loading in serialized model at each iteration")
    ("algo", po::value<string>(), "learning algorithms [ada|rmsprop|vsgd]")
    ("rho", po::value<double>(), "learning rate")
    ("tau", po::value<double>()->default_value(10), "window size")
    ("global.min_gamma_shape", po::value<double>(), "min gamma shape")
    ("iter", po::value<int>()->default_value(10), "number of iterations")
    ("test_interval", po::value<int>()->default_value(20), "interval for testing")
    ("samples", po::value<int>(), "samples for BBVI")
    ("max_examples", po::value<int>()->default_value(-1), "max examples used")
    ("seed", po::value<int>()->default_value(1234), "seed for random number generator")
    ("batch", po::value<int>()->default_value(-1), "batch size")
    ("threads", po::value<int>()->default_value(1), "OpenMP threads")
    ("batch_order", po::value<string>()->default_value("rand"), "seq|rand")
    ("train_load_params", po::value<string>(), "load params of train_model")
    ("model", po::value<string>(), "model INI file")
    ("double", po::value<bool>()->default_value(false), "Double DEF flag")
    ("predict_mode", po::value<bool>()->default_value(false), "Predict Mode")
    ("exp_fam_mode", po::value<bool>()->default_value(false), "Exponential Family Inference")
    ;

#define SAVE_OPT(vm, ptree, name, type)                         \
  if (vm.count(name)) ptree.put(name, vm[name].as<type>());
  
  auto save_options_to_ptree = [](const po::variables_map& vm,
                                  pt::ptree& ptree) {
    SAVE_OPT(vm, ptree, "v", int);
    SAVE_OPT(vm, ptree, "folder", string);
    SAVE_OPT(vm, ptree, "deserialization_test", bool);
    SAVE_OPT(vm, ptree, "algo", string);
    SAVE_OPT(vm, ptree, "rho", double);
    SAVE_OPT(vm, ptree, "global.min_gamma_shape", double);
    SAVE_OPT(vm, ptree, "tau", double);
    SAVE_OPT(vm, ptree, "iter", int);
    SAVE_OPT(vm, ptree, "test_interval", int);
    SAVE_OPT(vm, ptree, "samples", int);
    SAVE_OPT(vm, ptree, "max_examples", int);
    SAVE_OPT(vm, ptree, "seed", int);
    SAVE_OPT(vm, ptree, "batch", int);
    SAVE_OPT(vm, ptree, "threads", int);
    SAVE_OPT(vm, ptree, "batch_order", string);
    SAVE_OPT(vm, ptree, "train_load_params", string);
    SAVE_OPT(vm, ptree, "model", string);
    SAVE_OPT(vm, ptree, "double", bool);
    SAVE_OPT(vm, ptree, "predict_mode", bool);
    SAVE_OPT(vm, ptree, "exp_fam_mode", bool);
  };

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      exit(0);
    }
    LOG(info) << "reading config file";
    pt::ptree ptree;
    pt::ini_parser::read_ini(vm["model"].as<string>(), ptree);
    save_options_to_ptree(vm, ptree);
    return ptree;
  }
  catch(po::error e) {
    cout << e.what() << endl;
    exit(-1);
  }
} 

template<typename Model>
shared_ptr<Model> build_model (const string& model_type, pt::ptree* ptree) {
  auto options = ptree;
  // do batch training for validation & test
  if (model_type != "train") {
    options->put("batch", -1);
  }
  options->put("model_type", model_type);
  // init softmax shift
  init_shifted_softmax(*ptree);
  shared_ptr<Model> model(new Model(*options));
  return model;
};

// load model from deserialization, or if the data_file is present,
// build a model from the data file
template<typename Model>
shared_ptr<Model> try_and_make(const string& type, pt::ptree* ptree) {
  shared_ptr<Model> model;
  string deserialize_option = type + ".deserialize";
  string deserialize_file_option = type + ".deserialize_data_file";
  string file_option = type + ".data_file";

  if (ptree->get_optional<string>(deserialize_option)) {
    if (ptree->get_optional<string>(deserialize_file_option)) {
      // Try and use a new data path
      string data_file = ptree->get<string>(deserialize_file_option);
      LOG(info) << "Using new data for deserialized " << type
                << " model from " << data_file;
      model.reset(new Model(data_file));
    } else {
      model.reset(new Model());
    }
    string fname = ptree->get<string>(deserialize_option);
    LOG(info) << "Using deserialized " << type << " model from " << fname;
    deserialize_gzip<state_iarchive>(fname, model.get());
  } else if (ptree->get_optional<string>(file_option)) {
    model = build_model<Model>(type, ptree);

    // try to deserialize a partial model: we only use the layers up
    // to k from the deserialized model
    for(int k=0; k < 4; ++k) {
      if (ptree->get_optional<string>(type + ".deserialized_" + to_string(k) + "l")) {
        shared_ptr<Model> part_model(new Model());
        string part_fname = ptree->get<string>(type + ".deserialized_" + to_string(k) + "l");
        LOG(info) << "Using partial deserialized " << type << " model from " << part_fname;
        deserialize_gzip<state_iarchive>(part_fname, part_model.get());
        model->load_part(part_model, k);
      }
    }
  }
  if (type == "train")
    assert(model != NULL);
  return model;
}

template<typename Model>
void run_model(pt::ptree* ptree) {
  // build train & validation & test model
  shared_ptr<Model> train_model = try_and_make<Model>("train", ptree);
  shared_ptr<Model> valid_model = try_and_make<Model>("valid", ptree);
  shared_ptr<Model> test_model = try_and_make<Model>("test", ptree);

  auto max_iter = ptree->get<int>("iter");
  auto folder = ptree->get<string>("folder");

  // check iterations and save model
  auto save_model = [=](shared_ptr<Model>& model_ptr,
                        const string& model_type, bool final = false) {
    const Model& model = *model_ptr;
    auto i2s = [=](int d){
      char buffer[100];
      sprintf(buffer, "%05d", d);
      return string(buffer);
    };
    auto powerOf2 = [] (int i) {
      while (i > 1) {
        if (i / 2 * 2 != i) return false;
        i /= 2;
      }
      return true;
    };
    // save every 10 in the first 100 iterations, then every 200
    // iterations
    auto iteration = model.get_iteration();
    if (((iteration < 1000) && powerOf2(iteration))
        || (iteration % 1000 == 0) || final) {
      auto fname = folder + "/" + model_type + "_iter" +
        i2s(iteration) + ".archive.gz";
      auto param_fname = folder + "/" + model_type + "_iter" +
        i2s(iteration) + ".model.bin";
      serialize_gzip<state_oarchive>(fname, model);

      // Full serialization
      string full_fname = folder + "/" + model_type  + ".full.archive.gz";
      if ((iteration % 1000 == 0 && iteration > 0) || ptree->get<bool>("deserialization_test") || final) {
	model.set_full(true);
	serialize_gzip<state_oarchive>(full_fname, model);
	model.set_full(false);
      }

      model.save_params(param_fname);
      LOG(debug) << "serialize " << model_type << " at " << iteration;  

      // TEST Deserialization: 
      // You should get the same results while deserializing on the fly
      if (ptree->get<bool>("deserialization_test")) {
	shared_ptr<Model> model2(new Model());
	deserialize_gzip<state_iarchive>(full_fname, model2.get());
	model_ptr = model2;
      }
    }
  };

  auto test_interval = ptree->get<int>("test_interval");
  bool predict_mode = ptree->get<bool>("predict_mode");
  if (predict_mode) {
    valid_model->copy_w_params(*train_model);
    test_model->copy_w_params(*train_model);
  }

  for(int it=train_model->get_iteration(); it<max_iter; ++it) {
    auto t0 = time(NULL);
    printf("## Train\n");
    if (!predict_mode) {
      save_model(train_model, "train");
      train_model->train_model();

      valid_model->copy_w_params(*train_model);
      valid_model->copy_iteration(*train_model);

      test_model->copy_w_params(*train_model);
      test_model->copy_iteration(*train_model);
    }

    if (((it % test_interval == 0) || predict_mode) && (valid_model != NULL)) {
      printf("## Valid\n");
      save_model(valid_model, "valid");
      valid_model->train_model();
    }
    if (((it % test_interval == 0) || predict_mode) && (test_model != NULL)) {
      printf("## Test\n");
      save_model(test_model, "test");
      test_model->train_model();
    }

    auto t1 = time(NULL);
    printf("Iteration takes %d seconds\n", (int)(t1-t0));
  }

  save_model(train_model, "train", true);
  save_model(valid_model, "valid", true);
  save_model(test_model, "test", true);
}

int main(int argc, char* argv[]) {
  init_logging();
  LOG(info) << "def main running";
  LOG(info) << "def main parsing options";
  auto ptree = parse_options(argc, argv);
  init_logging(ptree);

  // rename the folder to include milisecond to make it unique
  {
    uint64_t milliseconds_since_epoch =
      std::chrono::system_clock::now().time_since_epoch() / 
      std::chrono::milliseconds(1);
    ptree.put("folder", ptree.get<string>("folder") + "_" + \
              to_string(milliseconds_since_epoch));
    set_output_folder(ptree.get<string>("folder"));
    cout << "model_folder " << ptree.get<string>("folder") << endl;
  }
  // create the folder for storing log and model
  {
    string cmd = "mkdir -p " + ptree.get<string>("folder");
    system(cmd.c_str());
  }
  // save the config file
  {
    string config_fname = ptree.get<string>("folder") + "/" + "config.ini";
    write_ini(config_fname, ptree);
  }
  if (ptree.get<bool>("double")) {
    run_model<DEF2Model>(&ptree);
  } else {
    cout << "HERE" << endl; 
    run_model<DEFModel>(&ptree);
  }
  return 0;
}

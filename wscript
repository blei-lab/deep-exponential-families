def options(opt):
  opt.load('compiler_cxx')
  opt.add_option('--mode', action='store', default='release', help='Compile mode: release or debug')

def configure(conf):
  print("configure!")

  # Set the compiler to waf
  conf.env.CXX = 'g++'
  conf.env.CC = 'gcc'
  conf.env.LIBPATH_MYLIB = ['/usr/lib', '/usr/lib64']

  if conf.options.mode == 'release':
    cxx_flags = ['-O3', '-fpermissive', '-std=c++11', '-march=native', '-DNDEBUG', '-fopenmp', '-DBOOST_LOG_DYN_LINK']
  else:
    cxx_flags = ['-O0', '-g', '-std=c++11', '-fopenmp', '-DBOOST_LOG_DYN_LINK']


  conf.env.append_value('CXXFLAGS', cxx_flags)

  conf.load('compiler_cxx')
  conf.check(compiler='cxx',lib=['m', 'gsl', 'gslcblas'], uselib_store='GSL')
  conf.check(compiler='cxx',lib='pthread', uselib_store='PTHREAD')
  conf.check(compiler='cxx',lib='gomp', uselib_store='OPENMP')
  conf.check(compiler='cxx',lib='armadillo', uselib_store='ARMADILLO')
  conf.check(compiler='cxx',lib='boost_program_options', uselib_store='PROGRAM_OPTIONS')
  conf.check(compiler='cxx',lib='boost_iostreams', uselib_store='IOSTREAMS')
  conf.check(compiler='cxx',lib='boost_serialization', uselib_store='SERIALIZATION')
  conf.check(compiler='cxx',lib='boost_filesystem', uselib_store='FILESYSTEM')
  conf.check(compiler='cxx',lib='boost_system', uselib_store='SYSTEM')
  conf.check(compiler='cxx',lib='boost_log', uselib_store='LOG')
  conf.check(compiler='cxx',lib='boost_random', uselib_store='RANDOM')

def build(bld):
  src = ['bbvi.cpp',
	'def_bernoulli_layer.cpp',
	'def.cpp',
	'def_data.cpp',
	'def_gamma_layer.cpp',
	'def_main.cpp',
	'def_model.cpp',
	'def_2_model.cpp',
	'def_poisson_y_layer.cpp',
	'def_poisson_layer.cpp',
	'def_gaussian_layer.cpp',
	'def_exponential_layer.cpp',
	'layer_factory.cpp',
	'link_function.cpp',
	'optimizer.cpp',
	'utils.cpp']

  lib = ['PTHREAD', 'ARMADILLO', 'PROGRAM_OPTIONS', 'IOSTREAMS', 'SERIALIZATION', 'FILESYSTEM', 'SYSTEM', 'OPENMP', 'GSL', 'LOG', 'RANDOM']
  bld.program(source=src, use=lib, target='def_main')


Deep Exponential Family
==========================

Reference
---------

[Deep Exponential Families](https://github.com/Blei-Lab/Publications/blob/master/2015_RanganathTangCharlinBlei/2015_RanganathTangCharlinBlei.pdf)
by Rajesh Ranganath, Linpeng Tang, Laurent Charlin, and David M. Blei, AISTATS 2015.


Requirements
------------

* armadillo
* boost 1.55
* OpenMP
* GSL
* g++ >= 4.7

Instructions to Build and Run
-----------------------------

Configuring: 
`./waf configure`

Building: 
`./waf build`
(binary is `build/def_main`)

Running: `def` reads its options from a config file and from the command
line. `./build/def_main --help` shows the command line options. 

We give a full example below (including a sample config file) of a def
running a dataset of wikipedia articles.

Input Format for Text
---------------------

The header:
```
n_examples n_words
```

Followed by `n_examples` examples, each has two lines:
```
example_ind example_words
word_ind0 word_count0 word_ind1 word_count1 ...
```


Comprehensive Example
---------------------

0. Data. We have pre-processed a corpus of wikipedia articles containing
1000 train articles, 500 validation and, 500 test articles. The data are
available in folder [wikpedia](wikipedia/)

0. Configuration file. The configuration file will be read by `def` and contains all
options regarding the model to be trained (e.g., number of layers, size of
layers, distribution of global and local variables).  Example
configuration for the wikipedia dataset is available in
folder [wikipedia](wikipedia/def_wikipedia_50_25_10.ini). Note that this file will look for the dataset in a directory specified by the WIKIPEDIA_DEF environment variable.

0. Running. Here is an example invocation:
```
cd deep-exponential-families
# define environment variable used in def_wikipedia_50_25_10.ini
export WIKIPEDIA_DEF=`pwd`/wikipedia
./build/def_main --v=3 --folder=experiments/def_wikipedia --algo=rmsprop --rho=.2 --samples=64 --max_examples=1000000 --model=wikipedia/def_wikipedia_50_25_10.ini --batch=10000 --batch_order=rand --threads=5 --test_interval=5 --iter=2000
```

The above settings (including the values provided in the configuration file)
are the ones we used in the paper. We have found these settings to be useful
across several datasets.


Topic visualization
-------------------

We show some of the tools that we have used to explore the def fits in this
[DEF IPython Notebook](http://nbviewer.ipython.org/github/Blei-Lab/deep-exponential-families/blob/master/wikipedia/def_wikipedia_visualization.ipynb).


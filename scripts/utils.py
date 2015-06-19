from pylab import *
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from collections import OrderedDict, namedtuple
from scipy.sparse import coo_matrix
import numpy
import os
import pandas

def softrect(x):
    t = -6
    return (x > t) * log(1+exp(x)) + (x <= t) * exp(x)

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def inv_nlf(x):
    return (x < 0) * (1.0/(1-x)) + (x >= 0) * (x+1.0)

def rect(x):
    return x * (x > 0)

def arg_type(v, ty=None):
    if isinstance(v, str):
        if ty is str:
            return v
        elif ty is None:
            return eval(v)
        else:
            return ty(v)
    else:
        return v

def load_text_matrix(fname, dtype=float64):
    doc_ind = []
    word_ind = []
    word_ct = []
    num_docs = 0
    num_words = 0
    with open(fname) as ifile:
        for l in ifile:
            wc_list = map(lambda p: map(int, p.split(':')), l.strip().split())[1:]
            if not wc_list:
                continue
            doc_ind.extend([num_docs] * len(wc_list))
            word_ind.extend(map(lambda (w, c): w, wc_list))
            word_ct.extend(map(lambda (w, c): c, wc_list))
            num_docs += 1
    words_doc = csc_matrix((word_ct, (word_ind, doc_ind)), dtype=dtype)
    return words_doc

def load_bin_model(fname, dtype=float64):
    with open(fname, mode='rb') as file: # b is
        # important -> binary
        bin_data = file.read()
    m = fromstring(bin_data, dtype=dtype)
    res = []
    i = 0
    while i < len(m):
        r, c = tuple(map(int, m[i:i+2]))
        res.append( reshape(m[i+2:i+2+r*c], (r, c)) )
        i += 2 + r*c
    return tuple(res)

def read_words(fname):
    words = []
    with open(fname) as ifile:
        for l in ifile:
            words.append(l.strip())
    return array(words)

def read_netflix_titles(fname):
    movies = [""] * 20000
    with open(fname) as ifile:
        for l in ifile:
            no, year, title = l.strip().split(',')[:3]
            movies[int(no)] = '"%s"' % title
    return array(movies)

def read_movielens_titles(fname):
    movies = [""] * 3953
    with open(fname) as ifile:
        for l in ifile:
            no, title, genre = l.strip().split('::')
            movies[int(no)] = title
    return array(movies)

def read_arxiv_titles(fname):
    papers = [""] * 20000
    with open(fname) as ifile:
        for i, l in enumerate(ifile):
	    if i >= 20000-1:
		break
            no, id, cat, title, date = l.strip().split('\t') #[:3]
	    papers[int(no)] = '"%s\t%s"' % (cat, title)
    return array(papers)

## arxiv double def visualization
from pandas import Series

# get the categories of papers that a user reads
def get_user_papers(u, arxiv, titles):
    papers_no = arxiv[u,:]
    # titles => categories => list of categories
    papers_read_cat = map(lambda t: t.strip('"').split('\t')[0].split(),
                          titles[papers_no > 0])
    # flatten the list
    cat_list = [c for c_list in papers_read_cat for c in c_list]
    cat_list = Series(cat_list)
    return cat_list

# get the representation of one user
# user_W: user x topics
# arxiv: user x papers
def get_user_topic(t, user_W, arxiv, titles, num_users=8, num_cat=4):
    top_users = argsort(user_W[:,t])[::-1][:num_users]
    top_users_cat = map(lambda u: list(get_user_papers(u, arxiv, titles).value_counts().index[:num_cat]),
                        top_users)
    return top_users_cat


def top_words(W, word_list, k=20, show_weight=True, topics_list=None, W_shape=None):
    def get_word_list(l):
        if isinstance(word_list, numpy.ndarray):
            return word_list[l]
        else:
            # word_list should be a function from word_ind to word
            return map(word_list, l)

    words, topics = W.shape
    if topics_list is None:
        topics_list = range(topics)
    result = OrderedDict()
    for j in topics_list:
        word_weight = W[:,j]
        word_ind = argsort(word_weight)[::-1]
        if show_weight:
            print 'Topic %s' % j
            if W_shape is None:
                print '\n'.join(map(lambda (w, x): '%s %s' % (w, x),
                                    zip(get_word_list(word_ind[:k]),
                                        word_weight[word_ind[:k]])))
            else:
                word_shape = W_shape[:,j]
                print '\n'.join(map(lambda (w, x, x2): '%s %s %s' % (w, x, x2),
                                    zip(get_word_list(word_ind[:k]),
                                        word_weight[word_ind[:k]],
                                        word_shape[word_ind[:k]])))
            print '\n'
        else:
            sep = ' ' if mean(map(lambda w: len(w), get_word_list(word_ind[:k]))) < 10 else '\n'
            print ('%02d ' % j) + sep + sep.join(get_word_list(word_ind[:k]))
        result[j] = zip(get_word_list(word_ind[:k]), word_weight[word_ind[:k]])
    return result

def top_groups(W1, W, word_list, k1=3, k=20, show_weight=True, groups_list=None):
    def get_word_list(l):
        if isinstance(word_list, numpy.ndarray):
            return word_list[l]
        else:
            # word_list should be a function from word_ind to word
            return map(word_list, l)

    topics, groups = W1.shape
    if groups_list is None:
        groups_list = range(groups)
    result = OrderedDict()
    # print W1
    word2groups = dot(W, W1)
    for j in groups_list:
        print 'group %d' % j
        topic_weight = W1[:,j]
        # print topic_weight
        topic_ind = argsort(topic_weight)[::-1]
        #print 'ok'
        #print topic_ind
        #print map(str, topic_ind)
        group_word_weight = word2groups[:,j]
        group_top_words = argsort(group_word_weight)[::-1]
        print ' '.join(get_word_list(group_top_words[:k]))
        print ' '.join(map(str, topic_ind[:k1]))
        print ' '.join(map(str, topic_weight[topic_ind[:k1]]))
        top_words(W, word_list, k, False, topics_list = topic_ind[:k1])
    return None

def top_supers(W2, W1, W, word_list, k2=3, k1=3, k=20, show_weight=True, super_list=None):
    def get_word_list(l):
        if isinstance(word_list, numpy.ndarray):
            return word_list[l]
        else:
            # word_list should be a function from word_ind to word
            return map(word_list, l)

    groups, supers = W2.shape
    if super_list is None:
        super_list = range(supers)
    result = OrderedDict()
    word2supers = dot(dot(W, W1), W2)
    # print W1
    for j in super_list:
        print 'SUPER %d' % j
        group_weight = W2[:,j]
        # print topic_weight
        group_ind = argsort(group_weight)[::-1]
        #print 'ok'
        #print topic_ind
        #print map(str, topic_ind)
        super_word_weight = word2supers[:,j]
        super_top_words = argsort(super_word_weight)[::-1]
        print ' '.join(get_word_list(super_top_words[:k]))

        print ' '.join(map(str, group_ind[:k2]))
        if show_weight:
            print ' '.join(map(str, group_weight[group_ind[:k2]]))
        top_groups(W1[:,group_ind[:k2]], W, word_list, k1, k, False)
        print '\n\n'
    return None

# sample gamma using matrices of shapes and scales
def sample_gamma(w_shape, w_scale):
    z = zeros_like(w_shape)
    for i in range(w_shape.shape[0]):
        for j in range(w_scale.shape[1]):
            z[i,j] = numpy.random.gamma(w_shape[i,j], w_scale[i,j])
    return z

# compute log p of gamma samples
def log_p_gamma(z, shape, scale):
    return -gammaln(shape) - shape*log(scale) + (shape-1)*log(z) - z/scale

def read_cpp_data(fname):
    row_list, col_list, value_list = [], [], []
    for k, line in enumerate(open(fname)):
        if k == 0:
            num_rows, num_cols = map(int, line.strip().split())
        elif k % 2 == 1:
            row_no, row_nnz = map(int, line.strip().split())
        else:
            ele = map(int, line.strip().split())
            for (col_no, ct) in zip(ele[::2], ele[1::2]):
                row_list.append(row_no)
                col_list.append(col_no)
                value_list.append(ct)
    return array(coo_matrix((value_list, (row_list, col_list)),
                            shape=(num_rows,num_cols)).todense())

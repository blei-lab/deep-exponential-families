import os
from pylab import argsort
from collections import OrderedDict, namedtuple
from StringIO import StringIO
from pprint import pprint
import pandas
import pyx
from utils import *
from pyx import *

class DEFHierarchy(object):
    def __init__(self):
        self.supers = OrderedDict()
        self.groups = OrderedDict()
        self.topics = OrderedDict()

    def __str__(self):
        sio = StringIO()
        sio.write("# Super Topics: %d\n" % len(self.supers))
        pprint(self.supers, sio)
        sio.write("\n# Groups: %d\n" % len(self.groups))
        pprint(self.groups, sio)
        sio.write("\n# Topics: %d\n" % len(self.topics))
        pprint(self.topics, sio)
        return sio.getvalue()

# result is a DEFHierarchy (supers, groups, topics)
# each supers/groups/topics is a ordered-dict:
# Each element in the list is a list of edges: (weight, node)
def gen_graph(W2, W1, W0, word_list, k2=3, k1=3, k0=20):
    result = DEFHierarchy()

    groups, supers = W2.shape
    for s in range(supers):
        gen_super(result, s, W2, W1, W0, word_list, k2, k1, k0)
    return result

def gen_graph_2layer(W1, W0, word_list, k1=3, k0=20):
    result = DEFHierarchy()

    topics, groups = W1.shape
    for g in range(groups):
        gen_group(result, g, W1, W0, word_list, k1, k0)
    return result

def gen_super(result, s, W2, W1, W0, word_list, k2, k1, k0):
    if s in result.supers:
        return result.supers[s]

    group_weight = W2[:,s]
    group_ind = argsort(group_weight)[::-1]
    for g in group_ind[:k2]:
        gen_group(result, g, W1, W0, word_list, k1, k0)

    super_node = [(g, W2[g,s]) for g in group_ind[:k2]]
    result.supers[s] = super_node
    return super_node

def gen_group(result, g, W1, W0, word_list, k1, k0):
    if g in result.groups:
        return result.groups[g]
    topic_weight = W1[:,g]
    topic_ind = argsort(topic_weight)[::-1]
    for t in topic_ind[:k1]:
        gen_topic(result, t, W0, word_list, k0)

    group_node = [(t, W1[t,g]) for t in topic_ind[:k1]]
    result.groups[g] = group_node
    return group_node

def gen_topic(result, t, W0, word_list, k0):
    if t in result.topics:
        return result.topics[t]

    word_weight = W0[:,t]
    word_ind = argsort(word_weight)[::-1]

    topic_node = [(word_list[w], W0[w,t]) for w in word_ind[:k0]]
    result.topics[t] = topic_node
    return topic_node

## -------------- drawing utilities ------------------

class DEFVisualizationLayer(object):
    def __init__(self):
        self.upper_connect = []
        self.lower_connect = []

def draw_layer(c, layer, marker_radius, y, x_center, x_space):
    vlayer = DEFVisualizationLayer()
    x_offset = x_center - (len(layer)-1) * x_space * 0.5
    for i, n in enumerate(layer):
        c.stroke(path.circle(i*x_space+x_offset, y, marker_radius), [style.linewidth.THICK])
        vlayer.upper_connect.append((i*x_space+x_offset, y+marker_radius))
        vlayer.lower_connect.append((i*x_space+x_offset, y-marker_radius))
    return vlayer

def draw_topic(c, words, y, x, text_size=text.size.LARGE):
    for i, w in enumerate(words):
        tbox = pyx.text.text(x, y-i, w, [text.halign.boxcenter, text_size])
        c.insert(tbox)
        pass

def draw_topics_layer(c, layer, y, x_center, x_space, text_size=text.size.LARGE):
    vl = DEFVisualizationLayer()
    x_offset = x_center - (len(layer)-1) * x_space * 0.5
    for i, n in enumerate(layer):
        words = map(lambda (w, e): w, layer[n])
        draw_topic(c, words, y, i*x_space+x_offset, text_size=text_size)
        vl.upper_connect.append((i*x_space+x_offset, y+0.5))
    return vl

# draw the lines from l1 to l2
def connect_layers(c, l1, l2, vl1, vl2):
    for i1, n1 in enumerate(l1):
        for (n2, w) in l1[n1]:
            i2 = l2.keys().index(n2)

            c.stroke(path.line(vl1.lower_connect[i1][0], vl1.lower_connect[i1][1],
                               vl2.upper_connect[i2][0], vl2.upper_connect[i2][1]),
                     [style.linewidth.THICK, deco.earrow(size=0.5)])

# draw groups and topics
def draw_groups(c, h, groups_space=5, topics_space=4, text_size=text.size.LARGE):
    vl_groups = draw_layer(c, h.groups, marker_radius=0.5, y=5, x_center=20, x_space=groups_space)
    vl_topics = draw_topics_layer(c, h.topics, y=0, x_center=20, x_space=topics_space, text_size=text_size)
    connect_layers(c, h.groups, h.topics, vl_groups, vl_topics)
    return c

# draw supers, groups and topics
def draw_supers(c, h, supers_space=6, groups_space=5, topics_space=4, text_size=text.size.LARGE):
    vl_supers = draw_layer(c, h.supers, marker_radius=0.7, y=10, x_center=20, x_space=supers_space)
    vl_groups = draw_layer(c, h.groups, marker_radius=0.5, y=5, x_center=20, x_space=groups_space)
    vl_topics = draw_topics_layer(c, h.topics, y=0, x_center=20, x_space=topics_space, text_size=text_size)
    connect_layers(c, h.supers, h.groups, vl_supers, vl_groups)
    connect_layers(c, h.groups, h.topics, vl_groups, vl_topics)
    return c

if __name__ == '__main__':
    t = map(softrect, load_bin_model(
        'experiments/nyt/gamma_20K/sparse/100_30_15/_1411792462847/train_iter15000.model.bin'))
    W0_shape, W0_scale, z0_shape, z0_scale = t[:4]
    z1_shape, z1_scale, z2_shape, z2_scale = t[4:8]
    W1_shape, W1_scale, W2_shape, W2_scale = t[8:]
    W0_mean = W0_shape * W0_scale
    W1_mean = W1_shape * W1_scale
    W2_mean = W2_shape * W2_scale

    h = gen_graph_2layer(W1_mean[:,[1,9,12]], W0_mean, word_list, 4, 8)
    c = canvas.canvas()
    draw_groups(c, h, groups_space=10)
    c.writePDFfile("groups_medical.pdf")

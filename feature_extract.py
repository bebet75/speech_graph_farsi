import numpy as np
from parsivar import *
import igraph
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from lines2words import *
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

def feature_extract_mw(test_string, patient, language = 'farsi'):

    stride = 100
    window_size = 400
    lines = test_string.split('.')
    coc_seq_list = []

    # Defining Counters
    sentence_counter = len(lines)
    coc_matrix = []
    seq_matrix = []
    seq_relations_matrix = {}

    if language == 'farsi':
        coc_relations_matrix, each_stemmed, word_counter_pro, word_counter = farsi_lines2words(lines)
    elif language == 'english':
        coc_relations_matrix, each_stemmed, word_counter_pro, word_counter = eng_lines2words(lines)
    else:
        print('No such language!')

    seq_nn = 0
    seq_ne = 0
    seq_ad = 0
    seq_awd = 0
    seq_ns = 0
    seq_gd = 0
    seq_aspl = 0
    seq_cc = 0
    seq_lq = 0
    clusters = 0
    lscc = 0
    seq_lscc = 0
    seq_diameter = 0
    seq_triangles = 0
    cooc_nn = 0
    cooc_ne = 0
    cooc_ns = 0
    cooc_ad = 0
    cooc_awd = 0
    cooc_gd = 0
    cooc_aspl = 0
    cooc_cc = 0
    cooc_lq = 0
    cooc_diameter = 0
    num_window = 0
    for i_start in range(0, len(each_stemmed), stride):
        i_end = i_start + window_size
        num_window = num_window + 1
        if i_end > len(each_stemmed):
            break
        # Producing Matrix of Co-occurrence Relations
        for each_edge in coc_relations_matrix.keys():
            each_relation = []
            each_relation.append(each_edge[0])
            each_relation.append(each_edge[1])
            each_relation.append(coc_relations_matrix[each_edge])
            coc_matrix.append(each_relation)
        # Producing Matrix of sequential relations
        for j in range(0, len(each_stemmed[i_start:i_end]) - 1):
            a = (each_stemmed[j], each_stemmed[j + 1])
            seq_relations_matrix[a] = seq_relations_matrix.get(a, 0) + 1

        for each_edge in seq_relations_matrix.keys():
            each_relation = []
            each_relation.append(each_edge[0])
            each_relation.append(each_edge[1])
            each_relation.append(seq_relations_matrix[each_edge])
            seq_matrix.append(each_relation)

        # Converting Sequential and Co-occurrence Matrices into Graphs
        g_cooc = igraph.Graph.TupleList(
            coc_matrix, weights=True, directed=False)
        g_seq = igraph.Graph.TupleList(seq_matrix, weights=True, directed=True)

        # Computing Graph Features
        seq_nn = g_seq.vcount() + seq_nn
        seq_ne = g_seq.ecount() + seq_ne
        seq_ns = np.mean(g_seq.strength())
        seq_ad = igraph.mean(g_seq.degree()) + seq_ad
        seq_awd = igraph.mean(g_seq.strength(weights='weight')) + seq_awd
        seq_gd = g_seq.density() + seq_gd
        seq_aspl = g_seq.average_path_length() + seq_aspl
        seq_cc = g_seq.transitivity_undirected() + seq_cc
        seq_lq = len(g_seq.largest_cliques()) + seq_lq
        clusters = g_seq.clusters()
        lscc = clusters.giant()
        seq_lscc = lscc.vcount() + seq_lscc
        seq_diameter = g_seq.diameter() + seq_diameter
        seq_triangles = len(g_seq.cliques(max=3, min=3)) + seq_triangles
        cooc_nn = g_cooc.vcount() + cooc_nn
        cooc_ne = g_cooc.ecount() + cooc_ne
        cooc_ns = np.mean(g_cooc.strength())
        cooc_ad = igraph.mean(g_cooc.degree()) + cooc_ad
        cooc_awd = igraph.mean(g_cooc.strength(weights='weight')) + cooc_awd
        cooc_gd = g_cooc.density() + cooc_gd
        cooc_aspl = g_cooc.average_path_length() + cooc_aspl
        cooc_cc = g_cooc.transitivity_undirected() + cooc_cc
        cooc_lq = len(g_cooc.largest_cliques()[0]) + cooc_lq
        cooc_diameter = g_cooc.diameter() + cooc_diameter

    seq_nn = seq_nn / num_window
    seq_ne = seq_ne / num_window
    seq_ad = seq_ad / num_window
    seq_awd = seq_awd / num_window
    seq_gd = seq_gd / num_window
    seq_aspl = seq_aspl / num_window
    seq_cc = seq_cc / num_window
    seq_lq = seq_lq / num_window
    seq_ns = cooc_ns / num_window
    seq_lscc = seq_lscc / num_window
    seq_diameter = seq_diameter / num_window
    seq_triangles = seq_triangles / num_window
    cooc_nn = cooc_nn / num_window
    cooc_ne = cooc_ne / num_window
    cooc_ad = cooc_ad / num_window
    cooc_awd = cooc_awd / num_window
    cooc_gd = cooc_gd / num_window
    cooc_aspl = cooc_aspl / num_window
    cooc_cc = cooc_cc / num_window
    cooc_lq = cooc_lq / num_window
    cooc_diameter = cooc_diameter / num_window
    cooc_ns = cooc_ns / num_window

    # Registering Graph Features into list
    coc_seq_features = []
    coc_seq_features.append(patient)
    coc_seq_features.append(word_counter)
    coc_seq_features.append(word_counter_pro / word_counter)
    coc_seq_features.append(word_counter_pro)
    coc_seq_features.append(sentence_counter)
    coc_seq_features.append(seq_nn)
    coc_seq_features.append(seq_ne)
    coc_seq_features.append(seq_ad)
    coc_seq_features.append(seq_awd)
    coc_seq_features.append(seq_gd)
    coc_seq_features.append(seq_aspl)
    coc_seq_features.append(seq_cc)
    coc_seq_features.append(seq_lq)
    coc_seq_features.append(seq_lscc)
    coc_seq_features.append(seq_diameter)
    coc_seq_features.append(seq_triangles)
    coc_seq_features.append(seq_ns)
    coc_seq_features.append(cooc_nn)
    coc_seq_features.append(cooc_ne)
    coc_seq_features.append(cooc_ad)
    coc_seq_features.append(cooc_awd)
    coc_seq_features.append(cooc_gd)
    coc_seq_features.append(cooc_aspl)
    coc_seq_features.append(cooc_cc)
    coc_seq_features.append(cooc_lq)
    coc_seq_features.append(cooc_diameter)
    coc_seq_features.append(cooc_ns)

    # Registering all features of an individual into list
    coc_seq_list.append(coc_seq_features)

    if word_counter < 400:
        print("Your list must have at least 400 words excluding preposition")

    return coc_seq_list


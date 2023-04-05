from parsivar import *
import igraph
from lines2words import *
from parsivar import *
import igraph
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')


def visualizer(test_string, language = 'farsi'):

    lines = test_string.split('.')
    coc_matrix = []
    seq_matrix = []
    seq_relations_matrix = {}

    if language == 'farsi':
        coc_relations_matrix, each_stemmed, word_counter_pro, word_counter = farsi_lines2words(lines)
    elif language == 'english':
        coc_relations_matrix, each_stemmed, word_counter_pro, word_counter = eng_lines2words(lines)
    else:
        print('No such language!')

    # Producing Matrix of Co-occurrence Relations
    for each_edge in coc_relations_matrix.keys():
        each_relation = []
        each_relation.append(each_edge[0])
        each_relation.append(each_edge[1])
        each_relation.append(coc_relations_matrix[each_edge])
        coc_matrix.append(each_relation)

    # Computing sequential relations
    for i in range(0, len(each_stemmed) - 1):
        a = (each_stemmed[i], each_stemmed[i + 1])
        seq_relations_matrix[a] = seq_relations_matrix.get(a, 0) + 1
    for each_edge in seq_relations_matrix.keys():
        each_relation = []
        each_relation.append(each_edge[0])
        each_relation.append(each_edge[1])
        each_relation.append(seq_relations_matrix[each_edge])
        seq_matrix.append(each_relation)

    # Converting Sequentail and Co-occurrence Matrices into the Graph
    g_cooc = igraph.Graph.TupleList(coc_matrix, weights=True, directed=False)
    g_seq = igraph.Graph.TupleList(seq_matrix, weights=True, directed=True)

    ############### Visualizing Co-occurrence Graph ##################
    for a in g_cooc.vs:
        a['vertex_label'] = a.index + 1
    visual_style = {}
    visual_style["vertex_label"] = g_cooc.vs["vertex_label"]
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"] = 10
    visual_style["vertex_label_color"] = 'black'
    visual_style["vertex_color"] = 'red'
    visual_style["edge_color"] = 'gray'
    # visual_style["edge_label"] = 'ho'
    visual_style["edge_label_size"] = 10
    visual_style["edge_label_color"] = 'blue'
    visual_style["layout"] = g_cooc.layout_fruchterman_reingold()
    visual_style["bbox"] = (800, 400)
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_label_angle"] = 0
    visual_style["margin"] = 100
    visual_style["edge_curved"] = 0.2

    print(g_cooc)
    igraph.plot(g_cooc, **visual_style)


########### Visualizaing Sequential Graph ############

    for a in g_seq.vs:
        a['vertex_label'] = a.index + 1
    visual_style = {}
    visual_style["vertex_label"] = g_seq.vs["vertex_label"]
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"] = 10
    visual_style["vertex_label_color"] = 'black'
    visual_style["vertex_color"] = 'red'
    visual_style["edge_color"] = 'gray'
    # visual_style["edge_label"] = g_seq.es["weight"]
    visual_style["edge_label_size"] = 5
    visual_style["edge_label_color"] = 'blue'
    visual_style["layout"] = g_seq.layout_fruchterman_reingold()
    visual_style["bbox"] = (600, 500)
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_label_angle"] = 0
    visual_style["margin"] = 60
    visual_style["edge_curved"] = 1

    igraph.plot(g_seq, **visual_style)

    for i in g_cooc.vs:
        print(i['name'], i.index + 1)

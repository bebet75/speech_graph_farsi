import numpy as np
from parsivar import *
import igraph
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def farsi_lines2words(lines):
    each_stemmed = []
    coc_relations_matrix = {}
    my_stemmer = FindStems()
    word_counter_pro = 0
    word_counter = 0

    for each in lines:
        word_list = each.strip().split(' ')
        unwanted = [
            'از',
            'که',
            'در',
            'ی',
            'می',
            'با',
            'بدون',
            'مانند',
            'بر',
            'درباره',
            'برای',
            'مثل',
            'را',
            'بی',
            'مگر'
            'به',
            'و',
            'تا',
            'هم',
            'چی',
            'چرا',
            'ولی',
            'ها',
            'هر',
            'یا',
            'اگر',
            'اگه',
            'نمی',
            'نم',
            'چون',
            'همراه',
            'روی']
        propositions = [
            'از',
            'که',
            'در',
            'با',
            'برای',
            'را',
            'بدون',
            'مانند',
            'درباره',
            'به',
            'و',
            'تا',
            'مثل',
            'هم',
            'مگر',
            'ولی',
            'هر',
            'یا',
            'اگر',
            'اگه',
            'چون',
            'بر',
            'بی',
            'همراه',
            'روی',
            ]
        for word in unwanted:
            word_list = list(filter((word).__ne__, word_list))
        for each_element_pro in propositions:
            if each_element_pro != '':
                word_counter_pro = word_counter_pro + 1
        for each_element in word_list:
            if each_element != '':
                word_counter = word_counter + 1

    # Stemming Words
        word_list_stemmed = []
        for each_word in word_list:
            if each_word != '':
                stem = my_stemmer.convert_to_stem(each_word)
                word_list_stemmed.append(stem)
    # Counting Co-occurrence Relations
        for first_element in range(0, len(word_list_stemmed) - 1):
            for second_element in range(
                    first_element + 1,
                    len(word_list_stemmed)):
                a = tuple(
                    sorted([word_list_stemmed[first_element], word_list_stemmed[second_element]]))
                coc_relations_matrix[a] = coc_relations_matrix.get(a, 0) + 1
    # Making Sequence of words
        for each_word in word_list:
            if each_word != '':
                stem = my_stemmer.convert_to_stem(each_word)
                each_stemmed.append(stem)
    return coc_relations_matrix, each_stemmed, word_counter_pro, word_counter


def eng_lines2words(lines):

    # Defining Counters
    word_counter = 0
    word_counter_pro = 0
    each_stemmed = []
    coc_relations_matrix = {}
    my_stemmer = FindStems()

    # Parsing into words
    for each in lines:
        word_list = []
        tokens = nltk.word_tokenize(each)
        sent = pos_tag(tokens, tagset='universal') 
        for (x,y) in sent:
            if y in ['DET','PRON','CONJ','.','PRT','ADP']:
                word_counter_pro = word_counter_pro+1
        sent_clean = [x for (x,y) in sent if y not in ['DET','PRON','CONJ','.','PRT','ADP']]
        for word in sent_clean:
            word_list.append(WordNetLemmatizer().lemmatize(word,'v'))

        for each_element in word_list:
            if each_element != '':
                word_counter = word_counter + 1

    # Stemming Words
        word_list_stemmed = []
        for each_word in word_list:
            if each_word != '':
                stem = my_stemmer.convert_to_stem(each_word)
                word_list_stemmed.append(stem)
    # Counting Co-occurrence Relations
        for first_element in range(0, len(word_list_stemmed) - 1):
            for second_element in range(
                    first_element + 1,
                    len(word_list_stemmed)):
                a = tuple(
                    sorted([word_list_stemmed[first_element], word_list_stemmed[second_element]]))
                coc_relations_matrix[a] = coc_relations_matrix.get(a, 0) + 1
    # Making Sequence of words
        for each_word in word_list:
            if each_word != '':
                stem = my_stemmer.convert_to_stem(each_word)
                each_stemmed.append(stem)
    
    return coc_relations_matrix, each_stemmed, word_counter_pro, word_counter

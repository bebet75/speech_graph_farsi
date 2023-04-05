from parsivar import Normalizer,Tokenizer ,POSTagger , FindStems

import nltk
import os


''' This Function inputs a persian text as a string and outputs the preprocessed text '''
def Persian_preprocessor(text):
    # assering the input is a string
    assert isinstance(text, str)
    
    # parsivar's normalizer on text
    normalized_text = norm_text(text)

    # replacing "---" and "..."  and other  punctuation marks
    text = remove_punct(normalized_text)

    # parsivar's POS tagging
    cleaned_tag_list = POS_preprocess(text)

    # parsivar's stemming
    stemmed_string = stemmer(cleaned_tag_list)

    return stemmed_string



'''this function inputs a string text and removes possible punctuation marks'''
def remove_punct(text):
    # assering the input is a string
    assert isinstance(text, str)
    # replacing "---" and "..."  and other punctuation marks with " "
    text = text.replace("---"," ")
    text = text.replace("-"," ")
    text = text.replace("..."," ")
    #text = text.replace("."," ")
    text = text.replace(","," ")
    text = text.replace(":"," ")
    text = text.replace(";"," ")
    text = text.replace("،"," ")
    text = text.replace("؛"," ")
    text = text.replace("!"," ")
    text = text.replace("?"," ")
    text = text.replace("؟"," ")
    text = text.replace("؟"," ")
    
    text = text.replace('\xad'," ")

    return text


'''this function only normalizes input string based on parsivar function'''
def norm_text(text):
    # assering the input is a string
    assert isinstance(text, str)

    normalizer = Normalizer()
    normalized_text = normalizer.normalize(text)
    return normalized_text


'''this function preprocesses input text string based on removing unwanted tags such as conjunctions
 then outputs a list of remaining words'''
def POS_preprocess(text):
    # assering the input is a string
    assert isinstance(text, str)

    tokenizer = Tokenizer()
    tagger = POSTagger(tagging_model="stanford")  # tagging_model = "wapiti" or "stanford". "wapiti" is faster than "stanford"
    text_tags = tagger.parse(tokenizer.tokenize_words(text))
    # Now removing unwanted tags from text
    cleaned_tag_list=[]
    for tag in text_tags:
        #removing words with tags = "con" ,"po", "ar"
        if tag[1]!='CON' and tag[1]!='PO' and tag[1]!='AR':
            cleaned_tag_list.append(tag[0])
    return cleaned_tag_list


'''This function stems the input list of words and outputs a text string '''
def stemmer(word_list):
    # assering the input is a list
    if type(word_list) is not list:
        raise ValueError('stemmer() input must be a list')

    stemmer = FindStems()
    stemmed_list=[]
    for word in word_list:
        stemmed_list.append(stemmer.convert_to_stem(word))
    # converting stemmed list to string
    stemmed_string = " ".join(stemmed_list)
    
    return stemmed_string




#java_path = "C:\Program Files\Java\jdk-18.0.1.1\\bin\java.exe"
#nltk.internals.config_java(java_path)
#os.environ['JAVAHOME'] = "C:\Program Files\Java\jdk-18.0.1.1\\bin\java.exe"
#def config_java(bin=java_path, options=None, verbose=False):
#    ...







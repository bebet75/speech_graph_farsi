# This Python file uses the following encoding: utf-8
from classifier import *
from data_to_feature import *
from feature_extract import *
from Plots import *
from visualizer import *
from forming_xy import *
import pandas as pd
import numpy as np
from parsivar import POSTagger
from parsivar import Tokenizer


def our_data_POS(speech_pts, Speech_controls, patients_info):
    df1 = pd.read_excel(speech_pts)
    df2 = pd.read_excel(Speech_controls)
    df = pd.concat([df1, df2], ignore_index=True)
    patient_dis = pd.read_excel(patients_info)
    control_dis = pd.read_excel(patients_info, sheet_name=3)
    patients = sorted(set(df["uid"].astype(str).tolist()))
    patients.remove('nan')
    control_dis['Subject Code'].astype(int).tolist()

    '''moving window! '''
    ### Network Analysis of Speech for whole data ###
    # Defining Stemmer and the list for registering variables
    PANSS = []
    diagnosis = []
    age = []
    education = []

    for index_pat, patient in enumerate(patients):
        whole_text = df[(df['uid'] == patient) & (
            df['speaker'] == 'subject') & (df['task'] == 'narration')]['content']
        text_list = whole_text.to_list()
        test_string = ' '.join([str(elem) for elem in text_list])
        coc_seq_list = feature_extract_POS(test_string)
        if index_pat == 0:
            dict_SOP = pd.DataFrame.from_dict(coc_seq_list,orient='columns')
        else:
            dict_SOP_new = pd.DataFrame.from_dict(coc_seq_list,orient='columns')
            dict_SOP = pd.concat([dict_SOP, dict_SOP_new], ignore_index=True, axis=0)

        # patient disorder
        patient_no = patient[1:]
        try:
            PANSS.append(patient_dis[(patient_dis['Subject Code'] == int(
                patient_no))]['Total PANSS score '].values[0])
        except IndexError:
            PANSS.append('nan')

        try:
            education.append(patient_dis[(patient_dis['Subject Code'] == int(
                patient_no))]['Years of Education'].values[0])
        except IndexError:
            education.append(control_dis[(control_dis['Subject Code'] == int(
                patient_no))]['Years of Education'].values[0])

        try:
            age.append(
                patient_dis[(patient_dis['Subject Code'] == int(patient_no))]['Age'].values[0])
        except IndexError:
            age.append(
                control_dis[(control_dis['Subject Code'] == int(patient_no))]['Age'].values[0])

        if int(patient_no) in control_dis['Subject Code'].astype(int).tolist():
            diagnosis.append('Control')
        if int(patient_no) in patient_dis['Subject Code'].astype(int).tolist():
            diagnosis.append('Schizo')

    # Converting List into a DataFrame
    dict_SOP.insert(1, 'PANSS', PANSS)
    dict_SOP.insert(2, 'diagnosis', diagnosis)
    dict_SOP.insert(3, 'age', age)
    dict_SOP.insert(4, 'education', education)
    dict_SOP.to_csv('features_mw.csv')
    print(dict_SOP)

    return dict_SOP


def feature_extract_POS(test_string, language = 'farsi'):
    
    lines = test_string.split('.')
    # Defining Counters
    dict_SOP = {}
    if language == 'farsi':
        word_counter_pro, word_counter = farsi_lines2words_forPOS(lines)
    elif language == 'english':
        pass
        # coc_relations_matrix, each_stemmed, word_counter_pro, word_counter = eng_lines2words(lines)
    else:
        print('No such language!')

    my_tokenizer = Tokenizer()
    my_tagger = POSTagger(tagging_model="wapiti")  # tagging_model = "wapiti" or "stanford". "wapiti" is faster than "stanford"
    text_tags = my_tagger.parse(my_tokenizer.tokenize_words(test_string))

    po = 0
    noun = 0
    adj = 0
    adv = 0
    v_pr = 0
    con = 0
    pro = 0

    for tup in text_tags:
        Pos = tup[1]
        if Pos == 'ADJ':
            adj = adj + 1
        if Pos == 'ADV':
            adv = adv + 1
        if Pos == 'N':
            noun = noun + 1
        if Pos == 'PO':
            po = po + 1
        if Pos == 'V_PR':
            v_pr = v_pr + 1
        if Pos == 'CON':
            con = con + 1
        if Pos == 'PRO':
            pro = pro + 1

    dict_SOP['ADJ'] = [adj/word_counter]
    dict_SOP['ADV'] = [adv/word_counter]
    dict_SOP['N'] = [noun/word_counter]
    dict_SOP['PO'] = [po/word_counter]
    dict_SOP['V_PR'] = [v_pr/word_counter]
    dict_SOP['CON'] = [con/word_counter]
    dict_SOP['PRO'] = [pro/word_counter]
    dict_SOP['PRO_code'] = [word_counter_pro/word_counter]

    return dict_SOP



def farsi_lines2words_forPOS(lines):
    word_counter_pro = 0
    word_counter = 0

    for each in lines:
        word_list = each.strip().split(' ')
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
        
        for each_element_pro in propositions:
            if each_element_pro != '':
                word_counter_pro = word_counter_pro + 1
        for each_element in word_list:
            if each_element != '':
                word_counter = word_counter + 1

    return word_counter_pro, word_counter


def forming_xy_POS(features, X_features):
    # Converting data (CSV) to dataframe

    # forming x and y
    wanted_features = ['diagnosis'] + X_features
    features_df = features[wanted_features]
    features_df = features_df.dropna(axis=0, how='any')
    X = features_df[X_features]
    Y = features_df['diagnosis'] == "Schizo"

    return X, Y, features_df


def forming_x(features, X_features):
    # Converting data (CSV) to dataframe
    features = features.drop(features[features['seq_nn'] == 0.0].index)
    # forming x and y
    features_df = features.dropna(axis=0, how='any')
    X = features_df[X_features]

    return X


X_features = ['PRO_code','N','ADJ','PO','ADV','PRO','V_PR','CON']

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(128,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(64,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(256,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(32,64)),
    LogisticRegression(solver='newton-cg', random_state=0),
    LogisticRegression(penalty='l1', solver='liblinear', random_state=0, max_iter = 200),
    LogisticRegression(random_state=0),
    LogisticRegression(penalty='l2', random_state=0)
]

speech_pts = 'speech-pts.xlsx'
Speech_controls = 'Speech-controls.xlsx'
patients_info = 'patients_info.xlsx'
dir = 'features_mw.csv'


'''
seq_df = our_data_POS(speech_pts,Speech_controls,patients_info)
X, Y, features_df = forming_xy_POS(seq_df,X_features)
classifier_full(X, Y, classifiers)
'''


'''
my_tokenizer = Tokenizer()
test_string = ' شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم عاشقانه ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم بلند انگلیسی  ا بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا ی شما بهترین متن عاشقانه بلند، متن بلند عاشقانه برای همسر، متن فوق العاده عاشقانه و تاثیرگذار، ناب ترین جملات احساسی بلند،متن عاشقانه بلند غمگین، متن عاشقانه بلند سالگرد ازدواج، متن عاشقانه بلند برای مخاطب خاص، متن عاشقانه بلند برای تولد عشقم، متن عاشقانه بلند انگلیسی و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم و دلنوشته های عاشقانه طولانی را گرد آوری کرده ایم، تا از آنها استفاده کنیداز آنها استفاده کنید'
my_tagger = POSTagger(tagging_model="wapiti")  # tagging_model = "wapiti" or "stanford". "wapiti" is faster than "stanford"
text_tags = my_tagger.parse(my_tokenizer.tokenize_words(test_string))
print(text_tags)
adj = 0
adv = 0
'''
import pandas as pd
import numpy as np
from parsivar import *
from feature_extract import *


def our_data_to_feature(speech_pts, Speech_controls, patients_info):
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
        coc_seq_list = feature_extract_mw(test_string, patient)

        if index_pat == 0:
            seq_df = pd.DataFrame(
                data=coc_seq_list,
                columns=[
                    'subject_id',
                    'word_count',
                    'Preposition ratio',
                    'Preposition',
                    'sentence_count',
                    'seq_nn',
                    'seq_ne',
                    'seq_ad',
                    'seq_awd',
                    'seq_gd',
                    'seq_aspl',
                    'seq_cc',
                    'seq_lq',
                    'seq_lscc',
                    'seq_diameter',
                    'seq_triangles',
                    'seq_ns',
                    'cooc_nn',
                    'cooc_ne',
                    'cooc_ad',
                    'cooc_awd',
                    'cooc_gd',
                    'cooc_aspl',
                    'cooc_cc',
                    'cooc_lq',
                    'cooc_diameter',
                    'cooc_ns'])
        else:
            seq_df_new = pd.DataFrame(
                data=coc_seq_list,
                columns=[
                    'subject_id',
                    'word_count',
                    'Preposition ratio',
                    'Preposition',
                    'sentence_count',
                    'seq_nn',
                    'seq_ne',
                    'seq_ad',
                    'seq_awd',
                    'seq_gd',
                    'seq_aspl',
                    'seq_cc',
                    'seq_lq',
                    'seq_lscc',
                    'seq_diameter',
                    'seq_triangles',
                    'seq_ns',
                    'cooc_nn',
                    'cooc_ne',
                    'cooc_ad',
                    'cooc_awd',
                    'cooc_gd',
                    'cooc_aspl',
                    'cooc_cc',
                    'cooc_lq',
                    'cooc_diameter',
                    'cooc_ns'])
            seq_df = pd.concat([seq_df, seq_df_new], ignore_index=True, axis=0)

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
    seq_df.insert(1, 'PANSS', PANSS)
    seq_df.insert(2, 'diagnosis', diagnosis)
    seq_df.insert(3, 'age', age)
    seq_df.insert(4, 'education', education)
    seq_df.to_csv('features_mw.csv')

    print(seq_df)
    return seq_df


def for_new_data(test_string, patient, language = 'farsi'):
    coc_seq_list = feature_extract_mw(test_string, patient, language = language)
    seq_df = pd.DataFrame(
        data=coc_seq_list,
        columns=[
            'subject_id',
            'word_count',
            'Preposition ratio',
            'Preposition',
            'sentence_count',
            'seq_nn',
            'seq_ne',
            'seq_ad',
            'seq_awd',
            'seq_gd',
            'seq_aspl',
            'seq_cc',
            'seq_lq',
            'seq_lscc',
            'seq_diameter',
            'seq_triangles',
            'seq_ns',
            'cooc_nn',
            'cooc_ne',
            'cooc_ad',
            'cooc_awd',
            'cooc_gd',
            'cooc_aspl',
            'cooc_cc',
            'cooc_lq',
            'cooc_diameter',
            'cooc_ns'])
    
    return seq_df


def for_new_dataset( list_patient, language = 'farsi'):
    list_patient = pd.read_excel(list_patient)
    diagnosis = []
    for patient_no in list_patient['Subject Code']:
        test_string = list_patient[list_patient['Subject Code'] == patient_no]['Text']
        seq_df = for_new_data(test_string, patient_no, language = language)
        diagnosis.append(list_patient[list_patient['Subject Code'] == patient_no]['Diagnosis'])
        seq_df.insert(2, 'Diagnosis', diagnosis)

    return seq_df
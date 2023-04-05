# Step 1: Import Libraries
import pandas as pd

dir = 'features_mw.csv'
'''
X_features = [
    'seq_ad',
    'seq_ns',
    'seq_awd',
    'seq_gd',
    'seq_aspl',
    'seq_cc',
    'seq_lq',
    'seq_lscc',
    'seq_diameter',
    'seq_triangles',
    'seq_ne',
    'seq_nn',
    'cooc_ad',
    'cooc_awd',
    'cooc_gd',
    'cooc_aspl',
    'cooc_cc',
    'cooc_lq',
    'cooc_diameter',
    'cooc_nn',
    'cooc_ne',
    'cooc_ns']
'''

def forming_xy(features, X_features):
    # Converting data (CSV) to dataframe
    features = features.drop(features[features['seq_nn'] == 0.0].index)

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
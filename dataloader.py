import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def preprocess_adult_data(attr = 'sex'):
    
    df0 = pd.read_csv('./adult/adult.data', header=None)
    df1 = pd.read_csv('./adult/adult.test', header=None, skiprows=[0])
    df = pd.concat([df0, df1], ignore_index=True)
    headers = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-stataus', 'occupation', 'relationship', 
        'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country', 'y'
    ]
    df.columns = headers
    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    FEATURES_CAT = ['workclass', 'education', 'marital-stataus', 'occupation', 'relationship', 
                    'race', 'sex', 'native-country']
    FEATURES_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    X_cat_raw = df.loc[:, FEATURES_CAT]
    X_num_raw = df.loc[:, FEATURES_NUM]
    y = df.y.values.reshape(-1, 1)

    if attr == 'race':
        enc_cat = OneHotEncoder(drop='if_binary', sparse=False)
    else:
        enc_cat = OneHotEncoder(sparse=False)
    enc_num = MinMaxScaler()
    X_cat = enc_cat.fit_transform(X_cat_raw)
    X_num = enc_num.fit_transform(X_num_raw)
    feat_names = enc_cat.get_feature_names_out()
    g_id = [i for i in range(len(feat_names)) if attr in feat_names[i]]

    X = np.concatenate((X_cat, X_num), 1)
    
    return X, y, g_id
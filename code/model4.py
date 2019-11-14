import sys

sys.path.append("./data/")
sys.path.append("./")
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pandas as pd

from nltk.tokenize import word_tokenize
from read_model_dic import read_moral_dic
import re

X_train_df = pd.read_pickle("X_train.pkl")
X_test_df = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl").to_numpy()
y_test = pd.read_pickle("y_test.pkl").to_numpy()

moral_dic = read_moral_dic()
keys = moral_dic.keys()


# key_names = ['HarmVirtue', 'AuthorityVirtue', 'PurityVirtue', 'HarmVice',
#              'PurityVice', 'IngroupVice',
#              'FairnessVirtue', 'MoralityGeneral', 'FairnessVice',
#              'IngroupVirtue', 'AuthorityVice']


def get_ratio(key, tokens):
    if len(tokens) == 0:
        return 0
    keywords = moral_dic[key]
    count = 0
    for word in keywords:
        r = re.compile(word)
        matches = list(filter(r.fullmatch, tokens))
        count += len(matches)
    return count / len(tokens)


def get_features(posts):
    X = []
    for text in posts:
        tokens = word_tokenize(text)
        sample = []
        for key in keys:
            sample.append(get_ratio(key, tokens))
        X.append(sample)
    return X


X_train = np.array(get_features(X_train_df['request_text']))
X_test = np.array(get_features(X_test_df['request_text']))

clf = SVC(kernel='linear')
clf.fit(X_train, y_train.ravel())

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC", roc_auc_score(y_test, y_pred))

import sys

sys.path.append("./data/")
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd

from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import roc_auc_score
import re

X_train_df = pd.read_pickle("X_train.pkl")
X_test_df = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl").to_numpy()
y_test = pd.read_pickle("y_test.pkl").to_numpy()


def get_keywords(filepath):
    f = open(filepath, "r")
    lines = f.readlines()
    words = [word.strip('\n') for word in lines]
    return words


def remove_symbols(text):
    return re.sub(r'[^A-Za-z \t]', '', text)


X_train_df['request_text'] = X_train_df['request_text'].apply(remove_symbols)
X_test_df['request_text'] = X_test_df['request_text'].apply(remove_symbols)

desire = get_keywords("./resources/narratives/desire.txt")
family = get_keywords("./resources/narratives/family.txt")
job = get_keywords("./resources/narratives/job.txt")
money = get_keywords("./resources/narratives/money.txt")
student = get_keywords("./resources/narratives/student.txt")
categories = [desire, family, job, money, student]


def get_features(posts):
    X = []
    for text in posts:
        tokens = word_tokenize(text)
        num_tokens = len(tokens)
        if num_tokens == 0:
            sample = [0] * len(categories)
            X.append(sample)
            continue
        counter = Counter(tokens)
        sample = []
        for category in categories:
            cat_count = 0
            for word in category:
                cat_count += counter[word]
            sample.append(cat_count / num_tokens)
        X.append(sample)
    return np.array(X)


X_train = get_features(X_train_df['request_text'])
X_test = get_features(X_test_df['request_text'])

clf = SVC(kernel='linear')
clf.fit(X_train, y_train.ravel())

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC", roc_auc_score(y_test, y_pred))

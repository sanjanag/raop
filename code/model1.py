import sys

import pandas as pd

sys.path.append("./data/")
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Load test and train sets
X_train_df = pd.read_pickle("X_train.pkl")
X_test_df = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl").to_numpy()
y_test = pd.read_pickle("y_test.pkl").to_numpy()

num_train = X_train_df.shape[0]


# vectorizers
corpus = list(X_train_df['request_text']) + list(X_test_df['request_text'])
uni_vec = CountVectorizer(ngram_range=(1, 1), strip_accents='ascii',
                          stop_words='english')
bi_vec = CountVectorizer(ngram_range=(2, 2), strip_accents='ascii',
                         stop_words='english')

# Vectorized text
X1 = uni_vec.fit_transform(corpus).toarray()
X2 = bi_vec.fit_transform(corpus).toarray()

# Find most frequent 500 bigrams and unigrams
uni_df = pd.DataFrame(X1, columns=uni_vec.get_feature_names())
bi_df = pd.DataFrame(X2, columns=bi_vec.get_feature_names())
freq_uni = list(uni_df.sum().sort_values(ascending=False)[:500].index)
freq_bi = list(bi_df.sum().sort_values(ascending=False)[:500].index)


# Construct X_train and X_test arrays
top_uni = uni_df[freq_uni].to_numpy()
top_bi = bi_df[freq_bi].to_numpy()
X = np.hstack((top_uni, top_bi))
X_train = X[:num_train]
X_test = X[num_train:]


# train_df = pd.DataFrame(X_train, columns=freq_uni + freq_bi)
# train_df['label'] = y_train

# Train classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train.ravel())

# Get predictions
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC", round(roc_auc_score(y_test, y_pred), 2))


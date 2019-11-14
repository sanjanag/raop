import sys

sys.path.append("./data/")
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score

X_train_df = pd.read_pickle("X_train.pkl")
X_test_df = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl").to_numpy()
y_test = pd.read_pickle("y_test.pkl").to_numpy()


def post_edited(obj):
    if obj == False:
        return 0
    return 1


def get_features(df):
    mod_df = df[['requester_account_age_in_days_at_request',
                 'requester_account_age_in_days_at_retrieval',
                 'requester_days_since_first_post_on_raop_at_request',
                 'requester_days_since_first_post_on_raop_at_retrieval',
                 'requester_number_of_comments_at_request',
                 'requester_number_of_comments_at_retrieval',
                 'requester_number_of_comments_in_raop_at_request',
                 'requester_number_of_comments_in_raop_at_retrieval',
                 'requester_number_of_posts_at_request',
                 'requester_number_of_posts_at_retrieval',
                 'requester_number_of_posts_on_raop_at_request',
                 'requester_number_of_posts_on_raop_at_retrieval',
                 'requester_number_of_subreddits_at_request',
                 'number_of_downvotes_of_request_at_retrieval',
                 'number_of_upvotes_of_request_at_retrieval',
                 'requester_upvotes_minus_downvotes_at_request',
                 'requester_upvotes_minus_downvotes_at_retrieval',
                 'requester_upvotes_plus_downvotes_at_request',
                 'requester_upvotes_plus_downvotes_at_retrieval'
                 ]].copy()
    mod_df['post_was_edited'] = df['post_was_edited'].apply(post_edited)
    enc = OneHotEncoder(categories=[['none', 'PIF', 'shroom']], drop='first')
    flair = enc.fit_transform(
        df['requester_user_flair'].fillna('none').to_numpy().reshape(-1,
                                                                     1)).toarray()
    X = mod_df.to_numpy()
    X = np.hstack((X, flair))
    return X


X_train = get_features(X_train_df)
X_test = get_features(X_test_df)

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train.ravel())

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC", round(roc_auc_score(y_test, y_pred), 2))

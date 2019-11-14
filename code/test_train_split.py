import sys

sys.path.append("./data/")
from data.read_dataset import read_dataset
import pandas as pd

dataset = read_dataset("./data/pizza_request_dataset.json")

df = pd.DataFrame(dataset)

y_df = df[['requester_received_pizza']]

df = df.drop(['requester_received_pizza'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y_df, test_size=0.1,
                                                    random_state=0)

X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")

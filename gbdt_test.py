import pandas as pd
import numpy as np

import lightgbm as lgb
import numpy as np

X = pd.read_csv("work/trn_curated_features.csv").drop(columns=["fname"])
Y = pd.read_csv("work/trn_curated_labels.csv").drop(columns=["fname"])
labels = list(Y.columns)
# print(labels)
# print(X.head())
# print(Y.head())
Y = Y[labels[2]]
# print(Y.shape)

X_tr = X[:1000]
X_te = X[1000:1500]
y_tr = Y[:1000]
y_te = Y[1000:1500]

train_data = lgb.Dataset(X_tr, label=y_tr)
valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'min_data_in_leaf': 30,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': 42,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, valid_data],
    valid_names=['train','valid'],
)

proba = bst.predict(X_te, num_iteration=bst.best_iteration)  # softmax 概率
print(proba.shape)
pred = proba > 0.02

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print("Accuracy:", accuracy_score(y_te, pred))
# print(classification_report(y_te, pred))
# print(confusion_matrix(y_te,pred))
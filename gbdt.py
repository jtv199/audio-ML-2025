import pandas as pd
import numpy as np
import lightgbm as lgb

from tqdm import tqdm
from sklearn.metrics import f1_score

class LightGBM():
    def __init__(
            self,
            n_labels,
            param
            ):
        self.n_labels = n_labels
        self.param = param
        self.bsts = []
        self.thresh = []
        self.search_steps = 1000
        
    def train(
            self,
            train_X,
            train_Y,
            val_X,
            val_Y
            ):
        del self.bsts
        del self.thresh
        self.bsts = []
        self.thresh = []
        pbar = tqdm(range(self.n_labels))
        for i in pbar:
            labels = list(train_Y.columns)
            train_data = lgb.Dataset(train_X, label=train_Y[labels[i]])
            valid_data = lgb.Dataset(val_X, label=val_Y[labels[i]], reference=train_data)
            bst = lgb.train(
                self.param,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, valid_data],
                valid_names=['train','valid'],
            )
            proba = bst.predict(train_X, num_iteration=bst.best_iteration)
            low = 0
            up = 0
            result = 0
            for j in range(self.search_steps+1):
                thresh = j / self.search_steps
                pred = proba > thresh
                this_result = f1_score(train_Y[labels[i]], pred)
                if this_result > result:
                    result = this_result
                    low = thresh
                elif this_result < result:
                    up = thresh - 1 / self.search_steps
                    break
            pos_num = sum(train_Y[labels[i]] == 1)
            threshold = low + (up - low) * (pos_num / len(train_Y))
            
            pred = bst.predict(val_X, num_iteration=bst.best_iteration) > threshold
            pbar.set_postfix(f1_score=f1_score(val_Y[labels[i]],pred))
            
            self.bsts.append(bst)
            self.thresh.append(threshold)

        

if __name__ == "__main__":
    X = pd.read_csv("work/trn_curated_features.csv").drop(columns=["fname"])
    Y = pd.read_csv("work/trn_curated_labels.csv").drop(columns=["fname"])

    param = params = {
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
    train_X = X[:1000]
    val_X = X[1000:1500]
    train_Y = Y[:1000]
    val_Y = Y[1000:1500]
    lgbm = LightGBM(80, param)
    lgbm.train(train_X,train_Y,val_X,val_Y)
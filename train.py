from dataset import AudioTabDataset
from gbdt import LightGBM
from gridsearch import GridSearchLGBM
from utils import lwlrap

import numpy as np

if __name__ == "__main__":
    outer = 10
    inner = 5
    ramdom_state = 42
    nclasses = 80
    params = {
        "objective": "binary",
        "metric": ("binary_logloss", "auc"),
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 30,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    table_path = "work/train_noisy_features.csv"
    label_path = "work/train_noisy_labels.csv"
    ds = AudioTabDataset(table_path, label_path, True)
    gs = GridSearchLGBM(80, params)
    ds.shuffle(ramdom_state)
    ds.kfold(outer)
    train_val, test = ds.train_test_split(0)
    train_val.kfold(inner)
    train, val = train_val.train_test_split(0)
    print(ds.n_samples, train.n_samples, val.n_samples, test.n_samples)
    print(ds.n_samples, train.n_samples + val.n_samples + test.n_samples)
    for i in range(outer):
        train_val, test = ds.train_test_split(i)
        test_X, test_Y = test.retrive()
        test_X = test_X.drop(columns=["fname"])
        test_Y = test_Y.drop(columns=["fname"])
        train_val.kfold(inner)
        for j in range(inner):
            train, val = train_val.train_test_split(j)
            train_X, train_Y = test.retrive()
            val_X, val_Y = test.retrive()
            train_X = train_X.drop(columns=["fname"])
            train_Y = train_Y.drop(columns=["fname"])
            val_X = val_X.drop(columns=["fname"])
            val_Y = val_Y.drop(columns=["fname"])
            models = gs.search(LightGBM, train_X, train_Y, val_X, val_Y)

            best_k = -1
            best_res = -1
            best_model = None
            best_param = None
            y_true = np.array(val_Y)
            for k in range(gs.size):
                model = models[k]
                y_prob, y_pred = model.pred(val_X)
                l = lwlrap(y_true, y_prob)
                if l > best_res:
                    best_res = l
                    best_k = k
                    best_model = model
                    best_param = gs.params[k]
        test_y_prob, test_y_pred = best_model.pred(test_X)
        print(lwlrap(test_y_prob, y_prob))

            




from itertools import product

from dataset import AudioTabDataset

class GridSearchLGBM():
    def __init__(self, nclasses, params):
        self.nclasses = nclasses
        self.params = self.grid_search_params(params)
        self.models = []
        self.size = len(self.params)

    def grid_search_params(self, param_dict):
        keys = list(param_dict.keys())
        values = []
        for k in keys:
            v = param_dict[k]
            if isinstance(v, list):
                values.append(v)
            else:
                values.append([v])
        combos = [dict(zip(keys, combo)) for combo in product(*values)]
        return combos
    
    def search(self, model, train_X, train_Y, val_X, val_Y):
        self.models = []
        for param in self.params:
            md = model(self.nclasses, param)
            md.train(train_X, train_Y, val_X, val_Y)
            self.models.append(md)
        return self.models
    
if __name__ == "__main__":
    params = {
        'objective': 'binary',
        'metric': ('binary_logloss', 'auc'),
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_data_in_leaf': 30,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': 42,
    }
    gs = GridSearchLGBM(80,params)

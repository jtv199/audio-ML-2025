import random
import numpy as np
import pandas as pd


class KFoldDataset:
    def __init__(self):
        self.files = []
        self.n_samples = 0
        self.folds = []
        self.k = 0

    def set_files(self, files):
        self.files = files
        self.n_samples = len(self.files)

    def shuffle(self, seed):
        rng = np.random.default_rng(seed)  # Independent RNG
        indices = np.arange(self.n_samples)
        rng.shuffle(indices)
        self.files = [self.files[i] for i in indices]

    def kfold(self, k):
        self.k = k
        # step = int(self.n_samples / self.k + 0.5)
        # for i in range(self.k):
        #     if i != k - 1:
        #         fold = self.files[step*i:step*(i+1)]
        #     else:
        #         fold = self.files[step*i:]
        #     self.folds.append(fold)
        self.folds = np.linspace(0, self.n_samples, self.k + 1, dtype=int)

    def get_file(self, n):
        return self.files[self.folds[n] : self.folds[n + 1]]
    
class AudioTabDataset(KFoldDataset):
    def __init__(
            self, 
            filepath=None,
            labelpath=None,
            readfile=True
            ):
        super().__init__()
        self.filepath=filepath
        self.labelpath=labelpath
        self.X = None
        self.Y = None
        if readfile:
            self.read()
    
    def read(self):
        self.X = pd.read_csv(self.filepath)
        self.Y = pd.read_csv(self.labelpath)
        self.set_files([i for i in range(len(self.X))])

    def train_test_split(self, n):
        train_indexes = []
        test_indexes = [idx for idx in range(self.folds[n],self.folds[n+1])]
        for i in range(self.k):
            if i == n:
                continue
            else:
                train_indexes += [idx for idx in range(self.folds[i],self.folds[i+1])]
        train_X = self.X.iloc[train_indexes]
        train_Y = self.Y.iloc[train_indexes]
        test_X = self.X.iloc[test_indexes]
        test_Y = self.Y.iloc[test_indexes]
        train_set = AudioTabDataset(readfile=False)
        test_set = AudioTabDataset(readfile=False)
        train_set.filepath = self.filepath
        train_set.labelpath = self.labelpath
        train_set.n_samples = len(train_X)
        train_set.X = train_X
        train_set.Y = train_Y
        test_set.filepath = self.filepath
        test_set.labelpath = self.labelpath
        test_set.n_samples = len(train_X)
        test_set.X = test_X
        test_set.Y = test_Y
        return train_set, test_set
    
    def retrive(self):
        return self.X, self.Y
    

if __name__ == "__main__":
    atfd = AudioTabDataset(readfile=False)
    n = 30
    k = 10
    X = pd.DataFrame({"x":[i for i in range(n)]})
    Y = pd.DataFrame({"y":[10*i for i in range(n)]})
    atfd.X = X
    atfd.Y = Y
    atfd.files = [i for i in range(n)]
    atfd.n_samples = n
    # atfd.shuffle(42)
    atfd.kfold(k)
    for i in range(k):
        print(f"Loop {k}")
        train, test = atfd.train_test_split(i)
        print(train.retrive())
        print(test.retrive())

    

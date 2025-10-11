import random
import numpy as np


class KFoldDataset():
    def __init__(self, files):
        self.files = files
        self.n_samples = len(self.files)
        self.folds = []
        self.k = 0
    
    def shuffle(self, seed):
        rng = np.random.default_rng(seed)   # 独立 RNG，不影响全局
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
    
    def get_fold(self, n):
        return self.files[self.folds[n]:self.folds[n+1]]


if __name__ == "__main__":
    kfd = KFoldDataset([i for i in range(10)])
    kfd.shuffle(42)
    k = 3
    kfd.kfold(k)
    for i in range(k):
        print(i, kfd.get_fold(i))
    

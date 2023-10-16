import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

def numpyNorm(X: np.ndarray):
    std = np.std(X, 0)
    mean = np.mean(X, 0)
    return np.nan_to_num(np.divide(np.subtract(X, mean), std)) 

def sklearnNorm(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


if __name__=="__main__":
    dataset = load_digits()
    np_X = numpyNorm(dataset['data'])
    skl_X = sklearnNorm(dataset['data'])
    print(np.array_equal(np_X, skl_X))

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

from scipy.spatial.distance import cdist
from sklearn.datasets import load_breast_cancer
from data_explorer import splitDataset, featuresOfDataset


class PerceptronModel:
    def __init__(self, input_shape:int):
        self.weights = torch.randn(size=(1, input_shape))

    def _update(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[0] == self.weights.shape[1], f'Shape mismatch! input shape of {input.shape[0]} with weights shape {self.weights.shape[1]}'
        assert Y in [-1, 1], 'unsupport target value!'
        
        y_hat = torch.sign((torch.matmul(self.weights, X.float())))
        #print(f'X: {X.shape} Y: {Y.shape}')

        if ~torch.eq(y_hat, Y):
            self.weights = torch.add(self.weights, Y*X.float().T)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):
        assert x_train.shape[0] == y_train.shape[0], 'Shape of train dataset mismatch!'
        assert x_val.shape[0] == y_val.shape[0], 'Shape of validation dataset mismatch!'

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def train(self, epoch: int = 1):
        assert epoch >= 1 and isinstance(epoch, int), 'Epoch must be a natural number!'
        
        training_cruve = []
        best_weights = self.weights
        for i in range(epoch):
            for x, y in zip(self.x_train, self.y_train):
                self._update(torch.tensor(x).reshape(-1 ,1), torch.tensor(y).reshape(1, -1))
            training_acc = self.eval(self.x_train, self.y_train)
            if len(training_cruve)>0 and training_acc > max(training_cruve): best_weights = self.weights
            training_cruve.append(training_acc)
        
        self.weights = best_weights
        return training_cruve
            
    def _predict(self, X: torch.Tensor):
        assert X.shape[1] == self.weights.shape[1], f'Shape mismatch! input shape of {X.shape[1]} with weights shape {self.weights.shape[1]}'
        return torch.sign((torch.matmul(self.weights, torch.from_numpy(X).T.float())))
            
    def eval(self, x_test: np.ndarray, y_test: np.ndarray):
        #evaluate the accuracy of the model within the test dataset
        #Error function: sum(abs(y_hat-y))/N
        assert x_test.shape[1] == self.weights.shape[1], f'Shape mismatch! input shape of {x_test.shape[1]} with weights shape {self.weights.shape[1]}'
        assert x_test.shape[0] == y_test.shape[0], 'Test data Shape mismatch!'
        assert set(y_test)==set([-1, 1]) , 'unsupport target value!'

        y_hat = self._predict(x_test)
        err = torch.sum(torch.abs(torch.subtract(y_hat, torch.from_numpy(y_test)))/2, dim=1)/y_test.shape[0]
        return float(1-err)
    
    def trainingCruvePlot(self, training_data: list, path: str = './plots/'):
        assert os.path.exists(path), f'The path: {path} does not exist!'

        epochs = list(range(1, len(training_data) + 1))

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, training_data, marker='o', linestyle='-')
        plt.title('Training Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(path, f'Training Curve'),  bbox_inches='tight')
        plt.close()
            
class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k = 1):
        assert k>0 and isinstance(k, int), 'K cannot be less than 0 and K must be an integer!'
        
        self.k = k

    def fit(self, X:np.ndarray, y:np.ndarray):
        assert X.shape[0] == y.shape[0], 'Train data Shape mismatch!'

        self.x_train = X
        self.y_train = y

    def _predict(self, datapoint: np.ndarray):
        assert isinstance(datapoint, np.ndarray), f'The data type {type(datapoint)}, it should be an instance of numpy.ndarray!'

        distances = cdist(self.x_train, [datapoint], metric='euclidean')
        k_indices = np.argsort(distances.squeeze())[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]
    
    def score(self, x_test: np.ndarray, y_test: np.ndarray):
        assert x_test.shape[0] == y_test.shape[0], 'Test data Shape mismatch!'

        predictions = np.array([self._predict(x) for x in x_test])
        err = np.sum(abs(predictions- y_test)/2)/y_test.shape[0]
        return 1-err

    def KNNGridSearchCV(self, K_max:int, cv:int = 5):
        assert K_max>0 and isinstance(K_max, int), 'K cannot be less than 0 and K must be an integer!'

        param_grid = {'k': [k+1 for k in range(K_max)]}
        grid_search = GridSearchCV(estimator=self, param_grid=param_grid, cv=cv)
        grid_search.fit(self.x_train, self.y_train)
        self.k = grid_search.best_params_['k']


def preprocessor(dataset, name:str = 'dataset'):
    datasetfeatures = {}
    featuresOfDataset(dataset['data'], dataset['target'], name, datasetfeatures , False)
    dataset['target'] = dataset['target']*2 -1
    return datasetfeatures

if __name__ == "__main__":
    PATH_TO_PLOTS = './Lab1/plots/'
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    dataset = load_breast_cancer()
    report = preprocessor(dataset)

    X_train, X_test, y_train, y_test = splitDataset(dataset['data'], 
                                                    dataset['target'], 
                                                    0.0, 
                                                    0.3, 
                                                    random_state = 42)
    K_MAX = 20
    CV = 10
    EPOCH = 20
    knn_model = KNN(k=1)
    K = knn_model.fit(X_train, y_train)
    knn_model.KNNGridSearchCV(K_max = K_MAX, cv = CV)

    print(f'Num of validation subsets: {CV}, maximum K in gridsearch: {K_MAX}')
    print(f'The best K of KNN: {knn_model.k}')
    print(f'Test accuracy of KNN: {knn_model.score(X_test, y_test)*100:.2f}% \n')


    X1_train, X1_val, X1_test, y1_train, y1_val, y1_test = splitDataset(dataset['data'], 
                                                                        dataset['target'], 
                                                                        0.2, 
                                                                        0.1, 
                                                                        random_state = 42)
    scaler = StandardScaler()
    X1_train = scaler.fit_transform(X1_train)

    preceptron = PerceptronModel(report['dataset']['input_shape'])
    preceptron.fit( X1_train, X1_val, y1_train, y1_val)
    trainingCruve = preceptron.train(EPOCH)
    preceptron.trainingCruvePlot(trainingCruve, PATH_TO_PLOTS)
    print(f'Num of epoch: {EPOCH}')
    print(f"Training accuracy of Perceptron: {preceptron.eval(X1_train, y1_train)*100:.2f}%")
    print(f"Validation accuracy of Perceptron: {preceptron.eval(X1_val, y1_val)*100:.2f}%")
    print(f"Test accuracy of Perceptron: {preceptron.eval(X1_test, y1_test)*100:.2f}%")
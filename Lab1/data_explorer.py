import os

import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_digits

from typing import Dict, Tuple

def featuresOfDataset(data: np.ndarray, targets: np.ndarray, NameOfDataset: str, DataCollector: Dict, FeaturesDetails: bool = True):
    assert isinstance(data, np.ndarray), f'The data type {type(data)}, it should be an instance of numpy.ndarray!'
    assert isinstance(targets, np.ndarray), f'The targets type {type(targets)}, it should be an instance of numpy.ndarray!'

    NumOfexamples = data.shape[0]
    input_shape = data.shape[1]
    labels, counts = np.unique(targets, return_counts=True)
    ExamplesDistr = counts/NumOfexamples*100

    collector = {}
    collector.update({'NumOfexamples': NumOfexamples})
    collector.update({'input_shape': input_shape})
    collector.update({'LabelValues': labels})
    collector.update({'ExamplesDistribution': ExamplesDistr})

    if FeaturesDetails:
        min_features = np.min(data, 0)
        max_features = np.max(data, 0)
        dev_features = np.std(data, 0)
        collector.update({'min_features': min_features})
        collector.update({'max_features': max_features})
        collector.update({'dev_features': dev_features})

    DataCollector.update({NameOfDataset: collector})

def findSimilarFeatures(data: np.ndarray):
    assert isinstance(data, np.ndarray), f'The data type {type(data)}, it should be an instance of numpy.ndarray!'

    datanorm = np.linalg.norm(data, axis=0)
    min_magnitude_diff = float('inf')
    similar_feature_indices = (-1, -1)

    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            magnitude_diff = abs(datanorm[i] - datanorm[j])
            if magnitude_diff < min_magnitude_diff:
                min_magnitude_diff = magnitude_diff
                similar_feature_indices = (i, j)

    return similar_feature_indices

def splitDataset(data: np.ndarray, target: np.ndarray, validation_ratio: float, test_ratio: float, random_state: int = 0):
    assert 0<validation_ratio+test_ratio<1.0, 'The sum of validation ratio and test ratio should be in range (0, 1.0)!'

    sumofratio = validation_ratio+test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(data, 
                                                        target, 
                                                        test_size=sumofratio, 
                                                        random_state=random_state)
    try:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, 
                                                        y_temp, 
                                                        test_size=test_ratio/sumofratio, 
                                                        random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    except:
        return X_train, X_temp, y_train, y_temp

#plot the dataset colored by labels
def plotFeatures(data: np.ndarray, NameOfDataset: str, indices: Tuple, target: np.ndarray, path: str = './plots/'):
    assert isinstance(data, np.ndarray), f'The data type {type(data)}, it should be an instance of numpy.ndarray!'
    assert os.path.exists(path), f'The path: {path} does not exist!'
    
    unique_labels = np.unique(target)
    colors = list(mcolors.BASE_COLORS.values())
    
    for label in unique_labels:
        label_data = data[target == label]
        plt.scatter(label_data[indices[0]], label_data[indices[1]], label=f'Class {label}', color=colors[label % len(colors)])

    plt.xlabel('first feature')
    plt.ylabel('second feature')
    plt.title(f'Similar features - {NameOfDataset}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(os.path.join(path, f'{NameOfDataset}-Similar_features.png'),  bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    PATH_TO_DATASETS = './dataset/'
    if not os.path.exists(PATH_TO_DATASETS):
        os.makedirs(PATH_TO_DATASETS)

    PATH_TO_PLOTS = './Lab1/plots/'
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    #Section 1, Q1(a); 
    #Wine reference: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    #Digits reference: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
    Datasets = {'Wine': load_wine(), 'Digits': load_digits()}
    DataCollector = {}
    for name, dataset in Datasets.items():
        featuresOfDataset(dataset['data'], 
                          dataset['target'], 
                          name, 
                          DataCollector)

    #Section 1, Q1(b), plot is saved at PATH_TO_PLOTS
    for name, dataset in Datasets.items():
        indices = findSimilarFeatures(dataset['data'])
        #plotFeatures(dataset['data'], name, indices, PATH_TO_PLOTS)
        plotFeatures(dataset['data'], name, indices, dataset['target'], PATH_TO_PLOTS)

    #Section 1, Q1(c); CIFAR10 reference: https://www.cs.toronto.edu/~kriz/cifar.html
    Datasets.update({'CIFAR10': torchvision.datasets.CIFAR10(PATH_TO_DATASETS, download=True)})
    featuresOfDataset(Datasets['CIFAR10'].data, 
                      np.array(Datasets['CIFAR10'].targets), 
                      'CIFAR10', 
                      DataCollector, 
                      False)
    print(DataCollector)

    #Section 1, Q2 
    X1_train, X1_val, X1_test, y1_train, y1_val, y1_test= splitDataset( Datasets['CIFAR10'].data, 
                                                                        np.array(Datasets['CIFAR10'].targets), 
                                                                        validation_ratio = 0.3, 
                                                                        test_ratio = 0.1, 
                                                                        random_state = 0)
    print(f'Train set shape: {X1_train.shape}; Validation set shape: {X1_val.shape}, test set shape: {X1_test.shape}')

    X2_train, X2_test, y2_train, y2_test = splitDataset(Datasets['Wine']['data'], 
                                                        Datasets['Wine']['target'], 
                                                        validation_ratio = 0, 
                                                        test_ratio = 0.3, 
                                                        random_state = 0)
    print(f'Train set shape: {X2_train.shape}; test set shape: {X2_test.shape}')
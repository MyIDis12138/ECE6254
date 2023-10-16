import os

import numpy as np
import matplotlib.pyplot as plt

def genData(dim: int, num:int = 50):
    return np.random.random((num, dim))-0.5

def genDataGuss(dim: int, num:int = 50):
    x = np.random.randn(num, dim)
    return np.clip(x, -0.5, 0.5)

def computeDisAng(data: np.ndarray):
    dist = []
    ang = []
    for i, d1 in enumerate(data):
        dist.append([])
        ang.append([])
        for d2 in data:
            dist[i].append(np.linalg.norm(d1-d2))
            cos = np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d1))
            ang[i].append(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.array(dist), np.array(ang)

def dataplt(dimOfData:int ,numOfData:int, dist: np.ndarray, ang: np.ndarray, path:str="./plots/"):
    for i in range(numOfData):
        for j in range(numOfData):
            plt.scatter(dist[i][j], ang[i][j])
    
    plt.xlabel('distance')
    plt.ylabel('angle(rad)')
    plt.savefig(os.path.join(path, f'Dim_of_{dimOfData}_Guss.png'),  bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    PATH_TO_PLOTS = "./Lab2/plots/"
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    DIM = [5, 50, 100]
    NUM = 50
    
    for dim in DIM:
        data = genDataGuss(dim, NUM)
        dist, ang = computeDisAng(data=data)
        dataplt(dim, NUM, dist, ang, PATH_TO_PLOTS)

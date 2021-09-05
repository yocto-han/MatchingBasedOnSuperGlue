import numpy as np
import csv
import pandas as pd
import torch

def arange_like(x, dim: int):
    # Cumsum ：计算轴向元素累加和，返回由中间结果组成的数组
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def compute_optimal_transport(M, r, c, lam = 27, epsilon = 1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))#行归r化，注意python中*号含义
        P *= (c / P.sum(0)).reshape((1, -1))#列归c化
    return P, np.sum(P * M)

if __name__ == '__main__':
    tmp_lst = []
    with open('TestData/1.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    X_category = data_0[['category']]
    X = data_0[['l1', 'l2', 'l3']]
    X = np.array(X, dtype=float)

    tmp_lst = []
    with open('TestData/2.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    Y_category = data_1[['category']]
    Y = data_1[['l1', 'l2', 'l3']]
    Y = np.array(Y, dtype=float)

    temp = [0] * 4
    descs1 = []
    for i in range(len(data_0)):
        if data_0["category"][i] == '1':
            temp[0] = 1
        elif data_0["category"][i] == '2':
            temp[1] = 1
        elif data_0["category"][i] == '3':
            temp[2] = 1
        elif data_0["category"][i] == '4':
            temp[3] = 1
        descs1.append(temp)
        temp = [0] * 4
    descs1 = np.array(descs1)

    temp = [0] * 4
    descs2 = []
    for i in range(len(data_1)):
        if data_1["category"][i] == '1':
            temp[0] = 1
        elif data_1["category"][i] == '2':
            temp[1] = 1
        elif data_1["category"][i] == '3':
            temp[2] = 1
        elif data_1["category"][i] == '4':
            temp[3] = 1
        descs2.append(temp)
        temp = [0] * 4
    descs2 = np.array(descs2)

    X = np.concatenate((X, descs1), axis=1)
    Y = np.concatenate((Y, descs2), axis=1)

    distribution_X = np.ones(len(X), )
    distribution_Y = np.ones(len(Y), )

    dist = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dist[i][j] = np.linalg.norm(X[i]-Y[j])



    P, d = compute_optimal_transport(dist, distribution_X, distribution_Y)
    P = torch.from_numpy(P)
    print(P)
    max0, max1 = P[:-1, :-1].max(0), P[:-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    print(indices1)
    P = P.numpy()
    P = pd.DataFrame(P)
    P.to_csv("P.csv")


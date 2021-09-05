#-*-coding:utf8-*-
import numpy as np
from data_utils import *
from hungarian import *
import csv
import time
import pandas as pd


def dist(f1, f2, weight):
    l2_dist = np.linalg.norm(f1[0:3]-f2[0:3])
    cls_dist = 1 - f1[3:].dot(f2[3:])
    ##apply weight
    distance = np.array([l2_dist, cls_dist]).dot(weight)
    return distance

def sinkhorn(M, r, c, lam, epsilon=1e-4, max_iter=5000):
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
    iter = 0
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
    # Hyper-parameter
    # dustbin代价
    dustbin_cost = 80
    # weight不同特征在计算cost时赋予的权重,
    # 目前是两个，一个是欧式距离的，一个是类别特征的
    weight = np.array([0.1, 0.9])
    # exponential decay
    exp_lambda = 0.1

    correct_n = 0
    total_matched_n = 0
    for m in range(1, 8):
        for n in range(m + 1, 8):
            path_1 = "TestData/" + str(m) + ".csv"
            path_2 = "TestData/" + str(n) + ".csv"

            tmp_lst = []
            with open(path_1, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_lst.append(row)
            data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            X_category = data_0[['category']]
            X = data_0[['l1', 'l2', 'l3']]
            X = np.array(X, dtype=float)

            tmp_lst = []
            with open(path_2, 'r') as f:
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

            features1 = np.concatenate((X, descs1), axis=1)
            features2 = np.concatenate((Y, descs2), axis=1)

            idx_0 = list(data_0['global_id'])
            idx_1 = list(data_1['global_id'])
            match_1 = []
            match_2 = []
            for idxx_0 in idx_0:
                if idxx_0 in idx_1:
                    match_1.append(idx_1.index(idxx_0))
                else:
                    match_1.append(len(data_1))
            for idxx_1 in idx_1:
                if idxx_1 in idx_0:
                    match_2.append(idx_0.index(idxx_1))
                else:
                    match_2.append(len(data_0))
            mid_1 = match_1
            mid_2 = match_2
            temp_1 = np.arange(len(data_0))
            temp_2 = np.arange(len(data_1))
            match_1 = np.hstack((temp_1, mid_2))
            match_2 = np.hstack((mid_1, temp_2))

            all_matches = set()
            for i in range(len(match_1)):
                all_matches.add((match_1[i], match_2[i]))

            dist_M = np.zeros((len(features1)+1,len(features2)+1))
            for i, f1 in enumerate(features1):
                for j, f2 in enumerate(features2):
                    dist_M[i,j] = dist(f1, f2, weight)
            ##
            dist_M[-1,:] = dustbin_cost
            dist_M[:,-1] = dustbin_cost
            dist_M[-1,-1] = 0

            ##Matching
            time_start = time.time()

            r = np.ones((len(features1)+1))
            r[-1] = len(features2)
            c = np.ones((len(features2)+1))
            c[-1] = len(features1)

            P, cost = sinkhorn(dist_M, r, c, exp_lambda)

            ##
            hungarian = Hungarian(P[0:-1,0:-1],is_profit_matrix=True)
            hungarian.calculate()
            time_end = time.time()
            ##
            results = hungarian.get_results()
            for pair in results:
                total_matched_n+=1
                if pair in all_matches:
                    correct_n+=1
    #
    print("原始数据匹配预测结果:正确匹配数/总匹配数 = {}/{}，准确率{}".format(correct_n, total_matched_n, correct_n / total_matched_n))

    correct_n = 0
    total_matched_n = 0
    for m in range(1, 8):
        for n in range(m + 1, 8):
            path_1 = "TestDataDesc/" + str(m) + ".csv"
            path_2 = "TestDataDesc/" + str(n) + ".csv"

            tmp_lst = []
            with open(path_1, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_lst.append(row)
            data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            X = data_0[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            features1 = np.array(X, dtype=float)

            tmp_lst = []
            with open(path_2, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_lst.append(row)
            data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            Y = data_1[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            features2 = np.array(Y, dtype=float)

            idx_0 = list(data_0['global_id'])
            idx_1 = list(data_1['global_id'])
            match_1 = []
            match_2 = []
            for idxx_0 in idx_0:
                if idxx_0 in idx_1:
                    match_1.append(idx_1.index(idxx_0))
                else:
                    match_1.append(len(data_1))
            for idxx_1 in idx_1:
                if idxx_1 in idx_0:
                    match_2.append(idx_0.index(idxx_1))
                else:
                    match_2.append(len(data_0))
            mid_1 = match_1
            mid_2 = match_2
            temp_1 = np.arange(len(data_0))
            temp_2 = np.arange(len(data_1))
            match_1 = np.hstack((temp_1, mid_2))
            match_2 = np.hstack((mid_1, temp_2))

            all_matches = set()
            for i in range(len(match_1)):
                all_matches.add((match_1[i], match_2[i]))

            dist_M = np.zeros((len(features1) + 1, len(features2) + 1))
            for i, f1 in enumerate(features1):
                for j, f2 in enumerate(features2):
                    dist_M[i][j] = np.linalg.norm(f1 - f2)
            ##
            dist_M[-1, :] = dustbin_cost
            dist_M[:, -1] = dustbin_cost
            dist_M[-1, -1] = 0

            ##Matching
            time_start = time.time()

            r = np.ones((len(features1) + 1))
            r[-1] = len(features2)
            c = np.ones((len(features2) + 1))
            c[-1] = len(features1)

            P, cost = sinkhorn(dist_M, r, c, exp_lambda)

            ##
            hungarian = Hungarian(P[0:-1, 0:-1], is_profit_matrix=True)
            hungarian.calculate()
            time_end = time.time()
            ##
            results = hungarian.get_results()
            for pair in results:
                total_matched_n += 1
                if pair in all_matches:
                    correct_n += 1
    #
    print("深度描述符匹配预测结果:正确匹配数/总匹配数 = {}/{}，准确率{}".format(correct_n, total_matched_n, correct_n / total_matched_n))

    correct_n = 0
    total_matched_n = 0
    for m in range(1, 8):
        for n in range(m + 1, 8):
            path_1 = "TestDataMLP/" + str(m) + ".csv"
            path_2 = "TestDataMLP/" + str(n) + ".csv"

            tmp_lst = []
            with open(path_1, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_lst.append(row)
            data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            X = data_0[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            features1 = np.array(X, dtype=float)

            tmp_lst = []
            with open(path_2, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_lst.append(row)
            data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            Y = data_1[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            features2 = np.array(Y, dtype=float)

            idx_0 = list(data_0['global_id'])
            idx_1 = list(data_1['global_id'])
            match_1 = []
            match_2 = []
            for idxx_0 in idx_0:
                if idxx_0 in idx_1:
                    match_1.append(idx_1.index(idxx_0))
                else:
                    match_1.append(len(data_1))
            for idxx_1 in idx_1:
                if idxx_1 in idx_0:
                    match_2.append(idx_0.index(idxx_1))
                else:
                    match_2.append(len(data_0))
            mid_1 = match_1
            mid_2 = match_2
            temp_1 = np.arange(len(data_0))
            temp_2 = np.arange(len(data_1))
            match_1 = np.hstack((temp_1, mid_2))
            match_2 = np.hstack((mid_1, temp_2))

            all_matches = set()
            for i in range(len(match_1)):
                all_matches.add((match_1[i], match_2[i]))

            dist_M = np.zeros((len(features1) + 1, len(features2) + 1))
            for i, f1 in enumerate(features1):
                for j, f2 in enumerate(features2):
                    dist_M[i][j] = np.linalg.norm(f1 - f2)
            ##
            dist_M[-1, :] = dustbin_cost
            dist_M[:, -1] = dustbin_cost
            dist_M[-1, -1] = 0

            ##Matching
            time_start = time.time()

            r = np.ones((len(features1) + 1))
            r[-1] = len(features2)
            c = np.ones((len(features2) + 1))
            c[-1] = len(features1)

            P, cost = sinkhorn(dist_M, r, c, exp_lambda)

            ##
            hungarian = Hungarian(P[0:-1, 0:-1], is_profit_matrix=True)
            hungarian.calculate()
            time_end = time.time()
            ##
            results = hungarian.get_results()
            for pair in results:
                total_matched_n += 1
                if pair in all_matches:
                    correct_n += 1
    #
    print("深度描述符(无Attention)匹配预测结果:正确匹配数/总匹配数 = {}/{}，准确率{}".format(correct_n, total_matched_n, correct_n / total_matched_n))

    def mul(A : list, num : float) -> list:
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = (A[i][j]) * num
        return A

    def mat_sum(A : list, B : list) -> list:
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] + B[i][j]
        return A


    def nomalize(A: list) -> list:
        max = np.max(A)
        min = np.min(A)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = (A[i][j] - min) / (max - min)
        return A


    diff_list = [(0.0 + i * 0.1) for i in range(11)]
    for diff in diff_list:
        correct_n = 0
        total_matched_n = 0
        for m in range(1, 8):
            for n in range(m + 1, 8):
                path_1 = "TestDataMLP/" + str(m) + ".csv"
                path_2 = "TestDataMLP/" + str(n) + ".csv"

                tmp_lst = []
                with open(path_1, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        tmp_lst.append(row)
                data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
                X = data_0[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                features1 = np.array(X, dtype=float)

                tmp_lst = []
                with open(path_2, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        tmp_lst.append(row)
                data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
                Y = data_1[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                features2 = np.array(Y, dtype=float)

                dist_M = np.zeros((len(features1) + 1, len(features2) + 1))
                for i, f1 in enumerate(features1):
                    for j, f2 in enumerate(features2):
                        dist_M[i][j] = np.linalg.norm(f1 - f2)
                ##
                dist_M[-1, :] = dustbin_cost
                dist_M[:, -1] = dustbin_cost
                dist_M[-1, -1] = 0

                path_1 = "TestData/" + str(m) + ".csv"
                path_2 = "TestData/" + str(n) + ".csv"

                tmp_lst = []
                with open(path_1, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        tmp_lst.append(row)
                data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
                X_category = data_0[['category']]
                X = data_0[['l1', 'l2', 'l3']]
                X = np.array(X, dtype=float)

                tmp_lst = []
                with open(path_2, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        tmp_lst.append(row)
                data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
                Y_category = data_1[['category']]
                Y = data_1[['l1', 'l2', 'l3']]
                Y = np.array(Y, dtype=float)

                idx_0 = list(data_0['global_id'])
                idx_1 = list(data_1['global_id'])
                match_1 = []
                match_2 = []
                for idxx_0 in idx_0:
                    if idxx_0 in idx_1:
                        match_1.append(idx_1.index(idxx_0))
                    else:
                        match_1.append(len(data_1))
                for idxx_1 in idx_1:
                    if idxx_1 in idx_0:
                        match_2.append(idx_0.index(idxx_1))
                    else:
                        match_2.append(len(data_0))
                mid_1 = match_1
                mid_2 = match_2
                temp_1 = np.arange(len(data_0))
                temp_2 = np.arange(len(data_1))
                match_1 = np.hstack((temp_1, mid_2))
                match_2 = np.hstack((mid_1, temp_2))

                all_matches = set()
                for i in range(len(match_1)):
                    all_matches.add((match_1[i], match_2[i]))

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

                features1 = np.concatenate((X, descs1), axis=1)
                features2 = np.concatenate((Y, descs2), axis=1)

                dist_M_temp = np.zeros((len(features1) + 1, len(features2) + 1))
                for i, f1 in enumerate(features1):
                    for j, f2 in enumerate(features2):
                        dist_M_temp[i, j] = dist(f1, f2, weight)
                #
                dist_M_temp[-1, :] = dustbin_cost
                dist_M_temp[:, -1] = dustbin_cost
                dist_M_temp[-1, -1] = 0

                dist_M = nomalize(dist_M)
                dist_M_temp = nomalize(dist_M_temp)
                score = mat_sum(mul(dist_M, diff), mul(dist_M_temp, (1 - diff)))



                ##Matching
                time_start = time.time()

                r = np.ones((len(features1) + 1))
                r[-1] = len(features2)
                c = np.ones((len(features2) + 1))
                c[-1] = len(features1)

                P, cost = sinkhorn(score, r, c, exp_lambda)

                ##
                hungarian = Hungarian(P[0:-1, 0:-1], is_profit_matrix=True)
                hungarian.calculate()
                time_end = time.time()
                ##
                results = hungarian.get_results()
                for pair in results:
                    total_matched_n += 1
                    if pair in all_matches:
                        correct_n += 1
        print("{:.2f}凸组合匹配预测结果:正确匹配数/总匹配数 = {}/{}，准确率{}".format(diff, correct_n, total_matched_n, correct_n / total_matched_n))









    


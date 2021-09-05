import numpy as np
import random
import time
import copy
import csv
import pandas as pd
from vote import check_global_id, get_global_id_of_instance
from collections import Counter
def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))
        if temp <= eps:
            N.append(i)
    return set(N)


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster

# MLP聚类方法
tmp_lst = []
with open('desc_V5.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
X = data_0[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
X = np.array(X, dtype = float)

eps = 2.6
min_Pts = 1
begin = time.time()
C = DBSCAN(X, eps, min_Pts)
data_0['label'] = C
grouped = data_0['global_id'].groupby(data_0['label'])
result = []
for group, data in grouped:
    data = np.array(data)
    for item in data:
        item = item + "#" + str(group)
        result.append(item)
    print(group)
    print(data)

group_merge_instance_dict, max_size_dict = {}, {}
group_global_id_dict_original, group_global_id_dict = {}, {}

for group, data in grouped:
    print(group)
    merge_instance_list, global_id_list = [], []
    for merge_instance in data:
        # task 1, 多个点融合为同一个实例
        # print(merge_instance)
        merge_instance_list.append(merge_instance)
        # task 2, 针对list中的每一个instance，获得该instance的global_id
        global_id = get_global_id_of_instance(merge_instance)
        # merge_instance_list中每个元素的对应global_id列表
        global_id_list.append(global_id) 
    group_merge_instance_dict[group] = merge_instance_list
    # task3, merge_instance_list中的所有global_id和对应的数量生成一个dict
    global_id_count = Counter(global_id_list)   
    global_id_dict = dict(global_id_count)   
    print("global_id_dict:", global_id_dict)
    # task4, 调用投票函数，获得max_size_global_id, max_size
    max_size_global_id, max_size = check_global_id(global_id_dict)
    group_global_id_dict_original[group] = [max_size_global_id, max_size]
    max_size_dict[group] = max_size
    # task5, 每个label都是独立的
    if max_size == -1:
        group_global_id_dict[group] = "unknown_50"
    elif max_size_global_id not in group_global_id_dict.values():
        group_global_id_dict[group] = max_size_global_id
    else:
        group_id = list(group_global_id_dict.keys())[list(group_global_id_dict.values()).index(max_size_global_id)]
        versus = lambda x,y:(x,y) if max_size_dict[x] > max_size_dict[y] else (y, x)
        winner, loser = versus(group_id, group)
        group_global_id_dict[loser] = "unknown_50"
        max_size_dict[loser] = -1
        group_global_id_dict[winner] = max_size_global_id
        
for group in group_global_id_dict_original.keys():
    print(group, group_global_id_dict_original[group], "——>", group, ":", group_global_id_dict[group], max_size_dict[group])

# task6, 实例维度的准确率, 此处疑惑: unknown怎么算？
total_merge_instance_size = 0
correct_merge_instance_size = 0
for group in group_merge_instance_dict.keys():
    for merge_instance in group_merge_instance_dict[group]:
        total_merge_instance_size += 1
        if get_global_id_of_instance(merge_instance) == group_global_id_dict[group]:
            correct_merge_instance_size += 1
        # elif group_global_id_dict[group] == "unknown_50":
        #     correct_merge_instance_size += 1
print("实例维度的准确率为：{:.4f}".format(correct_merge_instance_size/total_merge_instance_size))



result = pd.DataFrame(result)
print(result)
# result.to_csv("result.csv")
end = time.time()


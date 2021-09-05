import numpy as np
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset


class SparseDataset(Dataset):
    def __init__(self, data_path):
        self.data_files = []
        #
        # Prat1. 训练集由两部分构成，上部分为部分真实数据组合，这一部分训练 外点规模较大
        #
        # for i in range(20):
        #     for j in range(i+1, 20):
        #         temp = []
        #         temp.append(data_path + str(i) + ".csv")
        #         temp.append(data_path + str(j) + ".csv")
        #         self.data_files.append(temp)
        #
        # Part2. 这一部分为真实数据与据其制作的增广数据，这一部分外点规模较小
        #
        for i in range(172):
            temp = []
            temp.append(data_path + str(i) + ".csv")
            temp.append(data_path + str(i) + "warped.csv")
            self.data_files.append(temp)


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        tmp_lst = []
        global_id = []
        mat_0 = []
        mat_1 = []
        with open(self.data_files[idx][0], 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_lst.append(row)
                if row[4] != 'global_id':
                    global_id.append(row[4])

        data_0 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])

        tmp_lst = []
        with open(self.data_files[idx][1], 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_lst.append(row)
                if row[4] in global_id:
                    mat_0.append(global_id.index(row[4]))
                    mat_1.append(int(row[0]))
        data_1 = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])

        idx_0 = list(data_0['global_id'])
        idx_1 = list(data_1['global_id'])
        match_1 = []
        match_2 = []
        for idxx_0 in idx_0:
            if idxx_0 in idx_1:
                match_1.append(idx_1.index(idxx_0))
            else:
                match_1.append(-1)
        for idxx_1 in idx_1:
            if idxx_1 in idx_0:
                match_2.append(idx_0.index(idxx_1))
            else:
                match_2.append(-1)
        mid_1 = match_1
        mid_2 = match_2
        temp_1 = np.arange(len(data_0))
        temp_2 = np.arange(len(data_1))
        match_1 = np.hstack((temp_1, mid_2))
        match_2 = np.hstack((mid_1, temp_2))

        all_matches = []
        all_matches.append(np.array(match_1))
        all_matches.append(np.array(match_2))
        all_matches = np.array(all_matches)

        #
        # Part1. 初始描述符，由该实例对应的实例类别独热编码生产
        #
        # temp = [0] * 4
        # descs1 = []
        # for i in range(len(data_0)):
        #     if data_0["category"][i] == '1':
        #         temp[0] = 1
        #     elif data_0["category"][i] == '2':
        #         temp[1] = 1
        #     elif data_0["category"][i] == '3':
        #         temp[2] = 1
        #     elif data_0["category"][i] == '4':
        #         temp[3] = 1
        #     temp += temp
        #     descs1.append(temp)
        #     temp = [0] * 4
        # descs1 = np.array(descs1)
        #
        # temp = [0] * 4
        # descs2 = []
        # for i in range(len(data_1)):
        #     if data_1["category"][i] == '1':
        #         temp[0] = 1
        #     elif data_1["category"][i] == '2':
        #         temp[1] = 1
        #     elif data_1["category"][i] == '3':
        #         temp[2] = 1
        #     elif data_1["category"][i] == '4':
        #         temp[3] = 1
        #     temp += temp
        #     descs2.append(temp)
        #     temp = [0] * 4
        # descs2 = np.array(descs2)

        #
        # Part2. 改进后的描述符：实例类别信息拼接最近邻实例类别信息
        #
        descs1 = []
        x = data_0[['l1', 'l2', 'l3']]
        x = np.array(x, dtype = float)
        for i in range(len(data_0)):
            temp_0 = [0] * 4
            if data_0["category"][i] == '1':
                temp_0[0] = 1
            elif data_0["category"][i] == '2':
                temp_0[1] = 1
            elif data_0["category"][i] == '3':
                temp_0[2] = 1
            elif data_0["category"][i] == '4':
                temp_0[3] = 1
            min = 1000
            nearest = 0
            for j in range(len(data_0)):
                dist = np.sqrt(np.sum(np.square(x[i] - x[j])))
                if dist < min and j != i:
                    min = dist
                    nearest = j
            temp_1 = [0] * 4
            if data_0["category"][nearest] == '1':
                temp_1[0] = 1
            elif data_0["category"][nearest] == '2':
                temp_1[1] = 1
            elif data_0["category"][nearest] == '3':
                temp_1[2] = 1
            elif data_0["category"][nearest] == '4':
                temp_1[3] = 1
            temp = temp_0 + temp_1
            descs1.append(temp)
        descs1 = np.array(descs1)

        descs2 = []
        y = data_1[['l1', 'l2', 'l3']]
        y = np.array(y, dtype = float)
        for i in range(len(data_1)):
            temp_0 = [0] * 4
            if data_1["category"][i] == '1':
                temp_0[0] = 1
            elif data_1["category"][i] == '2':
                temp_0[1] = 1
            elif data_1["category"][i] == '3':
                temp_0[2] = 1
            elif data_1["category"][i] == '4':
                temp_0[3] = 1
            min = 1000
            nearest = 0
            for j in range(len(data_1)):
                dist = np.sqrt(np.sum(np.square(y[i] - y[j])))
                if dist < min and j != i:
                    min = dist
                    nearest = j
            temp_1 = [0] * 4
            if data_1["category"][nearest] == '1':
                temp_1[0] = 1
            elif data_1["category"][nearest] == '2':
                temp_1[1] = 1
            elif data_1["category"][nearest] == '3':
                temp_1[2] = 1
            elif data_1["category"][nearest] == '4':
                temp_1[3] = 1
            temp = temp_0 + temp_1
            descs2.append(temp)
        descs2 = np.array(descs2)

        temp = []
        kp1 = []
        for i in range(len(data_0)):
            temp.append(float(data_0['l1'][i]))
            temp.append(float(data_0['l2'][i]))
            temp.append(float(data_0['l3'][i]))
            kp1.append(temp)
            temp = []
        kp1_np = np.array(kp1)

        temp = []
        kp2 = []
        for i in range(len(data_1)):
            temp.append(float(data_1['l1'][i].strip("[]")))
            temp.append(float(data_1['l2'][i].strip("[]")))
            temp.append(float(data_1['l3'][i].strip("[]")))
            kp2.append(temp)
            temp = []
        kp2_np = np.array(kp2)

        # 调整特征点坐标格式
        kp1_np = kp1_np.reshape((1, -1, 3))
        kp2_np = kp2_np.reshape((1, -1, 3))

        scores1_np = torch.ones(len(data_0))
        scores2_np = torch.ones(len(data_1))

        id = []
        for i in range(len(data_0)):
            id.append(int(data_0['global_id'][i]))
        id = np.array(id)
        target = []
        for i in range(len(data_1)):
            target.append(int(data_1['global_id'][i]))
        target = np.array(target)

        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'all_matches': list(all_matches),
            'file_name': self.data_files[idx],
            'id': list(id),
            'target': list(target)
        }
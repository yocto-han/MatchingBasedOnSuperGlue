import numpy as np
import pandas as pd
import csv

data_path = 'data172/'
data_files = []
for i in range(172):
    tmp_lst = []
    with open(data_path + str(i) + ".csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    data = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    global_id = data['global_id']
    category = data['category']
    temp = []
    kp = []
    for j in range(len(data)):
        temp.append(float(data['l1'][j]))
        temp.append(float(data['l2'][j]))
        temp.append(float(data['l3'][j]))
        kp.append(temp)
        temp = []
    kp_np = np.array(kp)

    features_l1 = np.random.normal(scale = 0.5, loc = 0.0, size = (len(kp_np), 1))
    features_l2 = np.random.normal(scale = 0.3, loc = 0.0, size = (len(kp_np), 1))
    features_l3 = np.random.normal(scale = 0.0, loc = 0.0, size = (len(kp_np), 1))
    # 系统性误差引入
    # random_l1 = np.random.normal(loc = -0.12, scale = 7.61, size = None)
    # random_l2 = np.random.normal(loc = -0.01, scale = 10.52, size = None)
    # random_l3 = np.random.normal(loc = 0.00, scale = 0.00, size = None)
    # noise_l1 = random_l1 * (np.ones_like((kp_np[:, 0]).reshape(-1, 1)))
    # noise_l2 = random_l2 * (np.ones_like((kp_np[:, 1]).reshape(-1, 1)))
    # noise_l3 = random_l3 * (np.ones_like((kp_np[:, 2]).reshape(-1, 1)))
    # features_l1 += noise_l1
    # features_l2 += noise_l2
    # features_l3 += noise_l3
    # 噪声引入
    features_l1 += (kp_np[:, 0]).reshape(-1, 1)
    features_l2 += (kp_np[:, 1]).reshape(-1, 1)
    features_l3 += (kp_np[:, 2]).reshape(-1, 1)
    temp = zip(features_l1, features_l2, features_l3, global_id, category)
    temp = list(temp)
    name = ['l1', 'l2', 'l3', 'global_id', 'category']
    temp = pd.DataFrame(columns=name, data=temp)
    save_path = data_path + str(i) + "warped.csv"
    temp.to_csv(save_path, encoding='utf-8')
    #
    # # num = random.randrange(1, 5)
    # # if num == 3:
    # #     theta_yz = -15 * np.pi / 180
    # #     kp2_np = exchange_coor(kp2_np, theta_yz)
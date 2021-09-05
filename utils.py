import json
import random
import numpy as np
import math as m
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

def load_data(json_file):
    data = None
    with open(json_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)  # return a list, each element is a object
        return_data = []
        for object in data:
            if object["label_info"] != None:
                if object["label_info"]["location"] != None:
                    return_data.append(object)
        return return_data

def random_list(length):
    list = []
    for i in range(length):
        list.append(random.uniform(0.055, 0.070))
    return list

def transform(object):
    if object['prediction_info']['category'] == '红绿灯':
        object['prediction_info']['category'] = 1
    elif object['prediction_info']['category'] == '交通标志' or object['prediction_info']['category'] == '指路标志'\
            or object['prediction_info']['category'] == '指示标志':
        object['prediction_info']['category'] = 2
    elif object['prediction_info']['category'] == '杆':
        object['prediction_info']['category'] = 3
    elif object['prediction_info']['category'] == '转向箭头':
        object['prediction_info']['category'] = 4
    
    if object['label_info']['category'] == '红绿灯':
        object['label_info']['category'] = 1
    elif object['label_info']['category'] == '交通标志':
        object['label_info']['category'] = 2
    elif object['label_info']['category'] == '杆':
        object['label_info']['category'] = 3
    elif object['label_info']['category'] == '转向箭头':
        object['label_info']['category'] = 4

def warped() -> float:
    num = random.randrange(1, 3)
    if num == 2:
        return float(random.gauss(0, 1))
    else:
        return 0

def exchange_coor(pcd, theta_yz):
    point_arr = pcd
    ex_arr = np.array([[1, 0, 0], [0, m.cos(theta_yz), -m.sin(theta_yz)], [0, m.sin(theta_yz), m.cos(theta_yz)]])
    point_new = point_arr.dot(ex_arr)
    # pcd_new = o3d.geometry.PointCloud()
    # pcd_new.points = o3d.utility.Vector3dVector(np.transpose(point_new))
    # pcd_new = np.array(pcd_new.points)
    return point_new

def viz(matches, results, kpts0, kpts1):
    rdata_0, rdata_1 = kpts0, kpts1
    rmatches = matches
    rresults = results

    for j in range(len(kpts0)):
        data_0 = (rdata_0[j][0][0]).cpu().numpy()
        data_1 = (rdata_1[j][0][0]).cpu().numpy()
        matches = rmatches[j]
        results = rresults[j]
        x = []
        y = []
        p = []
        q = []
        for i in range(len(data_0)):
            x.append(data_0[i][0])
            y.append(data_0[i][1])
        for k in range(len(data_1)):
            p.append(data_1[k][0])
            q.append(data_1[k][1])

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(x, y, 'ko')
        ax2.plot(p, q, 'ko')

        true = [-1] * len(results)
        i = 0
        for result in results:
            for matche in matches:
                if np.array_equal(result, matche):
                    true[i] = 1
            i += 1

        for n in range(len(results)):
            if results[n][0] == -1:
                if true[n] == 1:
                    ax2.plot(p[results[n][1]], q[results[n][1]], 'go',markersize=5)
                else:
                    ax2.plot(p[results[n][1]], q[results[n][1]], 'ro',markersize=5)
            elif results[n][1] == -1:
                if true[n] == 1:
                    ax1.plot(x[results[n][0]], y[results[n][0]], 'go', markersize=5)
                else:
                    ax1.plot(x[results[n][0]], y[results[n][0]], 'ro', markersize=5)
            elif true[n] == 1:
                xy = (x[results[n][0]], y[results[n][0]])
                pq = (p[results[n][1]], q[results[n][1]])
                con = ConnectionPatch(xyA=xy, xyB=pq, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2,
                                      color="green")
                ax2.add_artist(con)
            elif true[n] == -1:
                xy = (x[results[n][0]], y[results[n][0]])
                pq = (p[results[n][1]], q[results[n][1]])
                con = ConnectionPatch(xyA=xy, xyB=pq, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2,
                                      color="red")
                ax2.add_artist(con)



        file_name = "result/" + str(j) + "line.png"
        plt.savefig(file_name)
        plt.cla()
        plt.close("all")
    print("OVER")

def confusion_matrix(P, N, TP, TN):
    classes = ['KeyPoints', 'DustbinPoints']
    confusion_matrix = np.array([(P, TP), (N, TN)], dtype=np.float64)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]))  # 显示对应的数字

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

def vote(test, matches):
    print(matches)
    for item in test:
        for matche in matches:
            if item[1] == str(matche[1]):
                item.append(str(matche[0]))
    vo = pd.DataFrame(test, columns = ['point', 'target', 'label'])
    print(vo)
    return 0
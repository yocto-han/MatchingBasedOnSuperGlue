import json
import pandas as pd
import os

file_path = "20210518/"
files = []
# os.listdir(train_path)会输出train_path下所有的文件的文件名
files += [file_path + f for f in os.listdir(file_path)]
l1 = []
l2 = []
l3 = []
for file in files:
    with open(file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    for object in data:
        if object['label_info']['global_id'] and object['label_info']['location']:
            l1.append(abs(object['label_info']['location'][0] - object['prediction_info']['location'][0]))
            l2.append(abs(object['label_info']['location'][1] - object['prediction_info']['location'][1]))
            l3.append(abs(object['label_info']['location'][2] - object['prediction_info']['location'][2]))
temp = zip(l1, l2, l3)
temp = list(temp)
name = ['l1', 'l2', 'l3']
csv = pd.DataFrame(columns=name, data=temp)
print(csv.describe())

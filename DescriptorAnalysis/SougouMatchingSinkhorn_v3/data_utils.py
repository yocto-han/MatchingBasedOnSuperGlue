#-*-coding:utf-8-*-
import json
from node import Node


def get_nodes_from_json_data(json_data):
    """
    load node from json data
    """
    nodes = []
    for node_data in json_data:
        ##skip some none data
        if node_data["prediction_info"]["location"] == None:
            continue
        if node_data["label_info"] == None:
            continue
        if node_data["label_info"]["location"] == None:
            continue
        ##extract data
        gid = node_data["label_info"]["global_id"]
        cls = node_data["prediction_info"]["category"]
        coor = node_data["label_info"]["location"]
        assert (gid is not None and cls is not None and coor is not None)
        #print(coor)
        ##
        n = Node(gid, coor, to_digit(cls))
        nodes.append(n)
    ##
    return nodes



def to_digit(v):
    #TL(红绿灯),P(杆),LM(转向箭头)
    if v=="交通标志" or v=="指路标志" or v=="指示标志":
        return 1
    if v=="红绿灯":
        return 2
    if v=="杆":
        return 3
    if v=="转向箭头":
        return 4
    return v


def load_raw_data(json_file):
    data = None
    with open(json_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def change_fmt(json_file):
    csv_data = []
    with open(json_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
        for obj in data:
            pred_info = [obj["prediction_info"]["instance_id"],
                          obj["prediction_info"]["category"]]
            if obj["prediction_info"]["location"] == None:
                pred_info.extend(["nan","nan","nan"])
            else:
                pred_info.extend(obj["prediction_info"]["location"])
            #
            label_info = []
            if obj["label_info"]== None:
                label_info.extend(["nan","nan","nan","nan","nan","nan"])
            else:
                label_info.extend([obj["label_info"]["global_id"],
                                  obj["label_info"]["instance_id"],
                                  obj["label_info"]["category"]])
                if obj["label_info"]["location"]==None:
                    label_info.extend(["nan","nan","nan"])
                else:
                    label_info.extend(obj["label_info"]["location"])
            csv_data.append((pred_info,label_info))
    return csv_data


if __name__=="__main__":
    # fpath = Path("./")
    # for fname in fpath.files("*.json"):
    #     csv_data = change_fmt(fname)
    #     with open(fname[:-5]+".csv","w",encoding="utf-8") as fout:
    #         for n_data in csv_data:
    #             line = ""
    #             for n in n_data:
    #                 for v in n:
    #                     v = to_digit(v)
    #                     line = line + str(v) +","
    #             line = line[:-1] + "\n"
    #             fout.write(line)
    # print("Done")

    data = load_raw_data("./2019-10-28-12-34-07_00_35_9E1FF29F-A3D6-449D-92AE-C989520B4986_000611-000725.json")
    nodes = get_nodes_from_json_data(data) 
    for n in nodes:
        print(n)
    print("Done")













    

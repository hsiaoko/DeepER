import pandas as pd
import numpy as np
import random


def train_data_generator():
    r_src =  "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/DBLP2.csv"
    l_src =  "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/ACM.csv"
    map_src =  "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/DBLP-ACM_perfectMapping.csv"

    l_data = pd.read_csv(l_src)
    r_data = pd.read_csv(r_src)
    map_data = pd.read_csv(map_src)

    super_num = 2
    df = pd.DataFrame()
    global_id=13081
    for index, row in l_data.iterrows():
        #print(row)
        l = row.values
        l_id = l[0]
        l = np.delete(l,0)
        for i in range(super_num):
            hash_val = random.uniform(0, len(r_data))
            r = r_data.loc[int(hash_val)]
            r = r.values
            r_id = r[0]
            r = np.delete(r, 0)

        
            labe_ = map_data.loc[map_data["idDBLP"] == r_id]
            labe_ = labe_.loc[labe_["idACM"] == l_id]
            label = 0
            if(labe_.empty):
                pass
            else:
                label = 1
                print(global_id, label, l_id, r_id, hash_val)
            tu_ = np.hstack((int(global_id), int(label), r, l))
            tu_ = pd.Series(tu_, name=global_id)
            print(tu_)
            df = df.append(tu_)
            global_id += 1
        #print(df)
        #break
    print(df)
    df.to_csv("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/dblp_acm_attr_5_2.csv", index=0)
    return

def ground_truth_generator(src_pth="/home/xiaoke/private/dataset/ER_Dataset/EC/TPACC/Test2005/csv/test_item_2005.csv"):

    src_data = pd.read_csv(src_pth)
    max_no = 10439821
    for i in range(1, max_no+1):
        #r = r_data.loc[int(hash_val)]
        match = src_data.loc[src_data['no'] == i]
        if len(match) > 1:
            print(match)
            break
        else:
            print(len(match), i)


    print(src_data)
if __name__ == '__main__':
    ground_truth_generator()

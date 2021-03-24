from distributed_rep import embeding
#from models import CLearnClearnERDataset
from models import core
import lsh_partition
import models
import torch
import pandas as pd
class MatchingModel():
    def __init__(self, embeding_style, embeding_source, schema, train_pt=None, eval_pt=None, prediction_l_pt=None, prediction_r_pt = None, args=None):
        self.train_data = pd.read_csv(train_pt)
        self.args = args
        if(embeding_style=="fasttext"):
            self.dm_data_set = core.DMFormatDataset(train_pt, embeding_source, 'avg', schema=schema)

    def run_train(self):
        for epoch in range(self.args['EPOCH']):
            tp = 0
            count =0
            for step, (x, y) in enumerate(self.dm_data_set):
        pass
    def run_prediction():
        pass
    def run_eval():
        pass


if __name__ == '__main__':
    schema = ['title', 'authors', 'venue', 'year']
    embeding_source = '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin'
    train_pt = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv"
    args = {
        "EPOCH":1,
        "BATCH_SIZE":64,
        "TIME_STEP":28,  
        "INPUT_SIEE":28,
        "LR":0.01
    }
    model = MatchingModel("fasttext", embeding_source, schema, train_pt = train_pt, args=args)
    model.run_train()
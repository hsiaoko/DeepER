import lsh_partition
import models
import torch
import pandas as pd
import numpy as np
import math
from distributed_rep import embeding
from models import core
from lsh_partition import lsh
from torch import nn
import time

class MatchingModel():
    def __init__(self, embeding_style, embeding_src, schema, train_src=None, eval_src=None, prediction_src=None, args=None, model_pt=None):
        self.args = args
        self.loss_reg = nn.SmoothL1Loss()
        self.loss_cls = nn.CrossEntropyLoss() 
        self.prediction_src = prediction_src
        self.model_pt = model_pt
        self.eb_model = embeding.FastTextEmbeding(source_pt=embeding_source)
        self.schema = schema
        if(embeding_style=="fasttext-avg"):
            self.model = core.ERModel()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['LR'])   # optimize all parameters
            self.dm_data_set_train = core.DMFormatDataset(train_src, self.eb_model, 'avg', schema=schema)
            self.dm_data_set_eval = core.DMFormatDataset(eval_src, self.eb_model, 'avg', schema=schema)
            

    def run_train_reg(self):
        for epoch in range(self.args['EPOCH']):
            tp = 0
            count =0
            self.optimizer.zero_grad()
            for step, (x, y) in enumerate(self.dm_data_set_train):
                out = self.model.forward(x)
                y = torch.reshape(y, (-1,1))
                #print(out, y)
                loss = self.loss_reg(out, y)
                if step % self.args['BATCH_SIZE'] == 0:
                    loss.backward()                 # backpropagation, compute gradients
                    self.optimizer.step()                # apply gradients
                if step % 100 == 0:
                    print("loss: ", loss)
                    self.optimizer.zero_grad()           # clear gradients for this training step
                    tp = 0
                    count = 0
            self.run_eval(0.1)
        sm = torch.jit.script(self.model)
        print("save: ", self.model_pt)
        sm.save(self.model_pt)
        return

    def run_train_cls(self):
        for epoch in range(self.args['EPOCH']):
            tp = 0
            count =0
            self.optimizer.zero_grad()
            for step, (x, y) in enumerate(self.dm_data_set_train):
                out = self.model.forward(x)
                loss = self.loss_cls(out, y)
                if step % self.args['BATCH_SIZE'] == 0:
                    loss.backward()                 # backpropagation, compute gradients
                    self.optimizer.step()                # apply gradients
                if step % 256 == 0:
                    print("Epoch: ", epoch, " - loss: ", loss)
                    self.optimizer.zero_grad()           # clear gradients for this training step
                    tp = 0
                    count = 0
            self.run_eval(eva_model='cls')
        sm = torch.jit.script(self.model)
        print("save: ", self.model_pt)
        sm.save(self.model_pt)
        return

    def run_eval(self, tau=None, eva_model='reg'):
        tp = 0 
        if eva_model=='reg':
            for step, (x, y) in enumerate(self.dm_data_set_eval):
                out = self.model.forward(x)
                y = torch.reshprediction_l_src.numpy()
                y = y.numpy()
                dis_ = x-y
                dis_ = np.reshape(dis_,-1)[0]
                dis_ = abs(dis_)
                if(dis_ < tau):
                    tp += 1
            print("prec: ", tp / self.dm_data_set_eval.length)
        elif eva_model =='cls':
            for step, (x, y) in enumerate(self.dm_data_set_eval):
                out = self.model.forward(x)
                out = torch.max(out, 1)[1].data.numpy()
                if(out[0] == y[0]):
                    tp += 1
        print("prec: ", tp/self.dm_data_set_eval.length)
        return

    def run_test(self):
        model = torch.jit.load(self.model_pt)
        if(type(self.prediction_src == tuple)):
            print(type(self.prediction_src))
            print(self.prediction_src)
            left_ = self.prediction_src[0]
            right_ = self.prediction_src[1]

        self.left_  = pd.read_csv(left_)
        self.right_  = pd.read_csv(right_)
        lsh_ = lsh.LSH(num_hash_func=1, num_hash_table=1, data_l=self.left_, data_r=self.right_, schema=self.schema, embeding_model=self.eb_model)
        lsh_.index()
        lsh_.show_hash_table()
        for table_id in range(lsh_.num_hash_table):
            table_ =  lsh_.get_table(table_id)
            for bucket_id in table_:
                for key_l in table_[bucket_id]:
                    eb_l = np.reshape(table_[bucket_id][key_l], (-1, 4, 100))
                    for key_r in table_[bucket_id]:
                        if (key_l[0] == key_r[0]):
                            continue
                        else:
                            eb_r = np.reshape(table_[bucket_id][key_r], (-1,4,100))
                            eb_ = np.concatenate((eb_l, eb_r), axis=0)
                            eb_ = torch.tensor(eb_, dtype=torch.float32)
                            out = self.model(eb_)
                            pred_y = torch.max(out, 1)[1].data.numpy()
                            if(pred_y == [0]):
                                continue
                            else:
                                print(key_l, key_r)
            #break

        return
    def run_prediction(self, tuple_l, tuple_r,schema):
        model = torch.jit.load(self.model_pt)
        
        eb_l = []
        eb_r = []
        for attr in schema:
            attr_l = "left_"+attr
            attr_r = "right_"+attr
            t_l = tuple_l[attr_l]
            t_r = tuple_r[attr_r]
            eb_l.append(self.eb_model.avg_embeding(str(t_l)))
            eb_r.append(self.eb_model.avg_embeding(str(t_r)))

        eb_l = np.reshape(eb_l, (-1, 4, 100))
        eb_r = np.reshape(eb_r, (-1, 4, 100))

        eb_ = np.concatenate((eb_l, eb_r), axis=0)
        eb_ = torch.tensor(eb_, dtype=torch.float32)
        out = model(eb_)

        pred_y = torch.max(out, 1)[1].data.numpy()
        print("pred: ", pred_y[0])
        if pred_y ==[0]:
            return False
        else:
            return True


if __name__ == '__main__':
    schema = ['title', 'authors', 'venue', 'year']
    embeding_source = '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin'
    train_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv"
    eval_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv"
    model_pt = "/home/LAB/zhuxk/project/DeepER/models/DBLP_ACM_classification.py"
    prediction_l_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/DBLP2.csv"
    prediction_r_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/ACM.csv"
    args = {
        "EPOCH":15,
        "BATCH_SIZE":16,
        "LR":0.001
    }
    model = MatchingModel("fasttext-avg", 
                        embeding_source, 
                        schema, 
                        train_src = train_src, 
                        eval_src = eval_src, 
                        args=args, 
                        model_pt=model_pt, 
                        prediction_src = (prediction_l_src, prediction_r_src)
                        )
    time_start=time.time()
    #model.run_train_cls()
    
    #model.run_test()

    data  = pd.read_csv(eval_src)
    tuple_l = data[['left_title','left_authors','left_venue', 'left_year']]
    tuple_r = data[['right_title','right_authors','right_venue', 'right_year']]

    tuple_l = tuple_l.iloc[1069]
    tuple_r = tuple_r.iloc[1069]
    print("TTTT:", tuple_l)

    print("TTTT:", tuple_r)
    model.run_prediction(tuple_l, tuple_r, schema)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')
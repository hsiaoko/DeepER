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
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class TrainModel():
    def __init__(self, embeding_style, embeding_src, schema, train_src=None, eval_src=None, train_args=None, model_pt=None, gt_src=None):
        assert model_pt != None and train_src != None and gt_src !=None and eval_src != None and train_args != None
        self.model_pt = model_pt
        self.eb_model = embeding.FastTextEmbeding(source_pt=embeding_src)
        self.schema = schema
        self.args = train_args
        self.model = core.ERModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['LR'])   # optimize all parameters
        
        self.gt_src = gt_src
        self.gt = pd.read_csv(gt_src)
        self.loss_reg = nn.SmoothL1Loss()
        self.loss_cls = nn.CrossEntropyLoss() 
        self.dm_data_set_train = core.DMFormatDataset(train_src, self.eb_model, 'avg', schema=schema)
        self.dm_data_set_eval = core.DMFormatDataset(eval_src, self.eb_model, 'avg', schema=schema)
        self.eva_model = EvalModel(self.dm_data_set_eval, self.gt, self.model, self.eb_model)
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
            #self.run_eval(0.1)
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
            self.eva_model.run_eval(eva_model = 'cls')
        sm = torch.jit.script(self.model)
        print("save: ", self.model_pt)
        sm.save(self.model_pt)
        return

    

class EvalModel():
    def __init__(self, dm_data_set_eval=None, gt=None, model=None, eb_model=None):
        #assert dm_data_set_eval != None and gt!=None, model!=None
        #assert eb_model !=None and model != None and gt != None and dm_data_set_eval !=None
        self.model = model
        self.eb_model = eb_model
        self.dm_data_set_eval = dm_data_set_eval
        self.gt = gt
    def run_eval(self, tau=None, eva_model='cls'):
        tp = 0 
        if eva_model=='reg':
            pass
        elif eva_model =='cls':
            for step, (x, y) in enumerate(self.dm_data_set_eval):
                out = self.model.forward(x)
                out = torch.max(out, 1)[1].data.numpy()
                if(out[0] == y[0]):
                    tp += 1
        print("prec: ", tp/self.dm_data_set_eval.length)
        return
    def get_f1(self,result):
        tp = 0
        print(self.gt)
        for row in result:
            data_slice = self.gt.loc[self.gt['idDBLP'] == row[0]]
            data_slice = data_slice.loc[data_slice['idACM'] == int(row[1])]
            if(data_slice.empty):
                continue
            else:
                tp+=1    
        print(tp/len(self.gt))
        return

class PredictModel():
    def __init__(self, embeding_style, embeding_src, schema, prediction_src=None, model_pt=None, gt_src=None, mpi_args = None, data=None, comm=None, size=None, rank=None):
        assert model_pt != None and embeding_style != None and embeding_src != None and schema != None
        self.model_pt = model_pt
        self.eb_model = embeding.FastTextEmbeding(source_pt=embeding_src)
        self.schema = schema
        self.prediction_src = prediction_src
        self.model = torch.jit.load(self.model_pt)
        self.eva_model = EvalModel()
        self.mpi_args = mpi_args
        self.data = data
        self.comm = comm
        self.rank = rank
        self.size = size
        self.data_ = []
        self.num_partitions = 0

    def data_partition(self):
        if(type(self.prediction_src == tuple)):
            print(type(self.prediction_src))
            print(self.prediction_src)
            left_ = self.prediction_src[0]
            right_ = self.prediction_src[1]

        self.left_  = pd.read_csv(left_)
        self.right_  = pd.read_csv(right_)
        lsh_ = lsh.LSH(num_hash_func=1, num_hash_table=1, data_l=self.left_, data_r=self.right_, schema=self.schema, embeding_model=self.eb_model)
        lsh_.index()
        #lsh_.show_hash_table()
        result = []
        if(self.mpi_args['role'] == 'master'):
            print(self.mpi_args)
            data_ = []
            for table_id in range(lsh_.num_hash_table):
                table_ =  lsh_.get_table(table_id)
                for bucket_id in table_:
                    data_.append(table_[bucket_id])
        self.num_partitions = len(data_)
        self.data_ = data_
        return 
        
    def get_part(self, part_id):
        return self.data_[part_id]

    def run_mpi_test(self, data_):
        print("run mpi_test: ", len(data_))
        result = []
        for key_l in data_:
            eb_l = np.reshape(data_[key_l], (-1, 4, 100))
            for key_r in data_:
                if(key_l[0]== 'R' and key_r[0]== 'L'):
                    continue
                if (key_l[0] == key_r[0]):
                    continue
                else:
                    eb_r = np.reshape(data_[key_r], (-1,4,100))
                    eb_ = np.concatenate((eb_l, eb_r), axis=0)
                    eb_ = torch.tensor(eb_, dtype=torch.float32)
                    out = self.model(eb_)
                    pred_y = torch.max(out, 1)[1].data.numpy()
                    #print("TEST:", "pred", pred_y, key_l, key_r, eb_)
                    if(pred_y == [0]):
                        continue
                    else:
                        print("pred: ", pred_y[0], key_l[2:], key_r[2:])
                        result.append([ key_l[2:], key_r[2:]])
        return
    def run_test(self):    
        if(type(self.prediction_src == tuple)):
            print(type(self.prediction_src))
            print(self.prediction_src)
            left_ = self.prediction_src[0]
            right_ = self.prediction_src[1]

        self.left_  = pd.read_csv(left_)
        self.right_  = pd.read_csv(right_)
        lsh_ = lsh.LSH(num_hash_func=1, num_hash_table=1, data_l=self.left_, data_r=self.right_, schema=self.schema, embeding_model=self.eb_model)
        lsh_.index()
        result = []
        for table_id in range(lsh_.num_hash_table):
            table_ =  lsh_.get_table(table_id)
            for bucket_id in table_:
                for key_l in table_[bucket_id]:
                    eb_l = np.reshape(table_[bucket_id][key_l], (-1, 4, 100))
                    for key_r in table_[bucket_id]:
                        if(key_l[0]== 'R' and key_r[0]== 'L'):
                            continue
                        if (key_l[0] == key_r[0]):
                            continue
                        else:
                            eb_r = np.reshape(table_[bucket_id][key_r], (-1,4,100))
                            eb_ = np.concatenate((eb_l, eb_r), axis=0)
                            eb_ = torch.tensor(eb_, dtype=torch.float32)
                            out = self.model(eb_)
                            pred_y = torch.max(out, 1)[1].data.numpy()
                            #print("TEST:", "pred", pred_y, key_l, key_r, eb_)
                            if(pred_y == [0]):
                                continue
                            else:
                                print("pred: ", pred_y[0], key_l[2:], key_r[2:])
                                result.append([ key_l[2:], key_r[2:]])

        print(len(result))
        self.eva_model.get_f1(result)
        return
    def run_test_without_blocking(self):
        if(type(self.prediction_src == tuple)):
            print(self.prediction_src)
            left_ = self.prediction_src[0]
            right_ = self.prediction_src[1]

        self.left_  = pd.read_csv(left_)
        self.right_  = pd.read_csv(right_)
        result = []
        for index_l, row_l in self.left_.iterrows():
            for index_r, row_r in self.right_.iterrows():
                eb_l = []
                eb_r = []
                for attr in self.schema:
                    t_l = row_l[attr]
                    t_r = row_r[attr]
                    eb_l.append(self.eb_model.avg_embeding(str(t_l)))
                    eb_r.append(self.eb_model.avg_embeding(str(t_r)))
                eb_l = np.reshape(eb_l, (-1,4,100))
                eb_r = np.reshape(eb_r, (-1,4,100))
                eb_ = np.concatenate((eb_l, eb_r), axis=0)
                eb_ = torch.tensor(eb_, dtype=torch.float32)
                out = self.model(eb_)
                pred_y = torch.max(out, 1)[1].data.numpy()
                print("FULL:","pred", pred_y, row_l["id"], row_r["id"])
        return

    def run_prediction(self, tuple_l, tuple_r,schema):
        #self.model = torch.jit.load(self.model_pt)
        
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
        out = self.model(eb_)

        pred_y = torch.max(out, 1)[1].data.numpy()
        print("pred: ", pred_y[0])
        if pred_y ==[0]:
            return False
        else:
            return True


class Runner():
    def __init__(self, mode=None, src_args=None, train_args=None, mpi_args = None):
        assert src_args != None 
        self.size = size
        self.rank = rank        
        self.mpi_args = mpi_args
        if(mode == "TRAIN"):
            assert train_args != None
            self.model = TrainModel(
                embeding_style = src_args['embeding_style'], 
                embeding_src = src_args['embeding_src'],
                schema = src_args['schema'],
                train_src = src_args['train_src'],
                eval_src = src_args['eval_src'], 
                train_args = train_args,
                model_pt = src_args['model_pt'],
                gt_src = src_args['gt_src'])

        elif(mode == 'PREDICT'):
            assert mpi_args != None 
            self.model = PredictModel(
                embeding_style = src_args['embeding_style'], 
                embeding_src = src_args['embeding_src'],
                schema = src_args['schema'],
                prediction_src = src_args['prediction_src'],                    
                model_pt = src_args['model_pt'],
                gt_src = src_args['gt_src'],
                mpi_args = self.mpi_args,
                comm = self.mpi_args['comm'],
                size = self.mpi_args['size'],
                rank = self.mpi_args['rank']
            )


if __name__ == '__main__':
    train_args = {
        "EPOCH":15,
        "BATCH_SIZE":16,
        "LR":0.0003
    }

    prediction_l_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/DBLP2.csv"
    prediction_r_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/ACM.csv"

    src_args = {
        "schema" : ['title', 'authors', 'venue', 'year'],
        "embeding_src" : '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin',
        "embeding_style" : 'fasttext',
        "train_src" : "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv",
        "eval_src" : "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv",
        "model_pt" : "/home/LAB/zhuxk/project/DeepER/models/DBLP_ACM_classification.py",
        "prediction_src" : (prediction_l_src, prediction_r_src),
        "gt_src" : "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/DBLP-ACM_perfectMapping.csv"
    }

    mpi_args = {
        "role" : "master",
        'num_workers' : 9,
        'worker_id' : 0,
        'comm' : comm,
        'rank' : rank,
        'size' : size
    }
    runner = Runner(src_args = src_args, mode="PREDICT", mpi_args = mpi_args)

    if rank == 0:

        #runner = Runner(src_args = src_args, train_args = train_args, mode="TRAIN")
        time_start=time.time()
        runner.model.run_train_cls()
        #runner.model.data_partition()
        #for i in range(runner.model.num_partitions):
        #    data_ = runner.model.get_part(i)
        #    comm.send(data_, dest=i+1)
        #runner.model.run_test_without_blocking()


        time_end=time.time()
        print('time cost',time_end-time_start,'s')
    else:
        print("A")
        #data_ = comm.recv()
        #runner.model.run_mpi_test(data_)
        #print("rank %d reveice : %s" % (rank, len(s)))
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
from torch.autograd import Variable


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
        self.model = core.ERModel(len(schema))
        #self.model = core.ERModelRNN()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['LR'])   # optimize all parameters
        
        self.gt_src = gt_src
        self.gt = pd.read_csv(gt_src)
        self.loss_reg = nn.SmoothL1Loss()
        self.loss_cls = nn.CrossEntropyLoss() 
        self.dm_data_set_train = core.DMFormatDataset(train_src, self.eb_model, 'avg', schema=schema)
        self.dm_data_set_eval = core.DMFormatDataset(eval_src, self.eb_model, 'avg', schema=schema)
        self.eva_model = EvalModel(self.dm_data_set_eval, self.gt, self.model, self.eb_model, schema=self.schema)
    """

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
    """

    def run_train_cls(self):

        train_loader = torch.utils.data.DataLoader(dataset=self.dm_data_set_train, batch_size=self.args['BATCH_SIZE'], shuffle=True)
        for epoch in range(self.args['EPOCH']):
            self.optimizer.zero_grad()
            #for step, (x, y) in enumerate(self.dm_data_set_train):
            for step, (x, y) in enumerate(train_loader):        # gives batch data
                out_ = self.model.forward(x)
                out = torch.max(out_, 1)[1].data
                out = out.reshape((-1, 1))
                ccc = out - y
                tp = torch.tensor(np.array([0]), dtype=torch.float32,requires_grad = True)
                p_ = torch.tensor(np.array([0]), dtype=torch.float32,requires_grad = True)
                y = torch.tensor(y, dtype=torch.float32, requires_grad=True)

                #for i in range(y.shape[0]):
                #    tp = torch.add(tp, 1)
                #    print(tp)
                c = y[:] >= 1 
                b = out[:]>=1
                ss = torch.sum(c)
                sss = torch.sum(c&b)
                loss1 = torch.tensor(ss-sss, dtype=torch.float32,requires_grad = True)

                out_ = torch.reshape(out_, (-1,2))
                y = y.reshape((-1))
                loss2 = self.loss_cls(out_, y.long())                    
                loss = loss2

                if step % self.args['BATCH_SIZE'] == 0:
                    loss.backward()
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
    def __init__(self, dm_data_set_eval=None, gt=None, model=None, eb_model=None, schema=None):
        #assert dm_data_set_eval != None and gt!=None, model!=None
        #assert eb_model !=None and model != None and gt != None and dm_data_set_eval !=None
        self.model = model
        self.eb_model = eb_model
        self.dm_data_set_eval = dm_data_set_eval
        self.gt = gt
        self.schema = schema
    def run_eval(self, tau=None, eva_model='cls'):
        tp = 0 
        if eva_model=='reg':
            pass
        elif eva_model =='cls':
            for step, (x, y) in enumerate(self.dm_data_set_eval):

                x = np.reshape(x, (1,2,len(self.schema),100))
                out = self.model.forward(x)
                out = torch.max(out, 1)[1].data.numpy()
                print(y[0], out[0])
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
        self.gt = pd.read_csv(gt_src)
        self.eva_model = EvalModel(gt=self.gt, schema=self.schema)
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
            eb_l = np.reshape(data_[key_l], (-1, len(self.schema), 100))
            for key_r in data_:
                if(key_l[0]== 'R' and key_r[0]== 'L'):
                    continue
                if (key_l[0] == key_r[0]):
                    continue
                else:
                    eb_r = np.reshape(data_[key_r], (-1,len(self.schema),100))
                    eb_ = np.concatenate((eb_l, eb_r), axis=0)
                    eb_ = np.reshape(eb_, (1,2,len(self.schema),100))
                    eb_ = torch.tensor(eb_, dtype=torch.float32)
                    out = self.model(eb_)
                    pred_y = torch.max(out, 1)[1].data.numpy()
                    #print("TEST:", "pred", pred_y, key_l, key_r)
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
                    eb_l = np.reshape(table_[bucket_id][key_l], (-1, len(self.schema), 100))
                    for key_r in table_[bucket_id]:
                        if(key_l[0]== 'R' and key_r[0]== 'L'):
                            continue
                        if (key_l[0] == key_r[0]):
                            continue
                        else:
                            eb_r = np.reshape(table_[bucket_id][key_r], (-1,len(self.schema),100))
                            eb_ = np.concatenate((eb_l, eb_r), axis=0)
                            eb_ = np.reshape(eb_, (1,2,len(self.schema),100))
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
                eb_l = np.reshape(eb_l, (-1,len(self.schema),100))
                eb_r = np.reshape(eb_r, (-1,len(self.schema),100))
                eb_ = np.concatenate((eb_l, eb_r), axis=0)
                eb_ = np.reshape(eb_, (1,2,len(self.schema),100))
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

        eb_l = np.reshape(eb_l, (-1, len(schema), 100))
        eb_r = np.reshape(eb_r, (-1, len(schema), 100))

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
        "EPOCH":30,
        "BATCH_SIZE":64,
        "LR":0.001
    }

    prediction_l_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/DBLP2.csv"
    prediction_r_src = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/ACM.csv"

    src_args = {
        "schema" : ['title','authors','venue','year'],
        "embeding_src" : '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin',
        "embeding_style" : 'fasttext',
        "train_src" : "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/dblp_acm_attr_5_1.csv",
        "eval_src" : "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/dblp_acm_attr_10_1.csv",
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
    runner = Runner(src_args = src_args, train_args = train_args, mode="TRAIN")
    
    #runner = Runner(src_args = src_args, mode="PREDICT", mpi_args = mpi_args)

    time_start=time.time()

    if rank == 0:

        runner.model.run_train_cls()
       # runner.model.data_partition()
       # for i in range(runner.model.num_partitions):
       #     data_ = runner.model.get_part(i)
       #     comm.send(data_, dest=i+1)
        #runner.model.run_test_without_blocking()
        #runner.model.run_test()



    else:
        print("A")
        #data_ = comm.recv()
        #runner.model.run_mpi_test(data_)
        #print("rank %d reveice : %s" % (rank, len(s)))
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
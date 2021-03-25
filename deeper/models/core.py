import copy
import logging
from collections import Mapping
import dill
import six
#import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from distributed_rep import embeding as eb
import pandas as pd
import numpy as np


class ERModel(nn.Module):
    def __init__(self):
        super(ERModel, self).__init__()

        self.rnn = nn.LSTM(     
            input_size=100,      # 图片每行的数据像素点
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.hidden = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)    # 输出层
        self.out_cls = nn.Linear(16, 2)    # 输出层

    def forward(self, x):
        c = torch.split(x, 1, dim = 0)
        x_l = c[0]
        x_r = c[1]

        sim_rep = x_l-x_r
        #sim_rep = torch.tensor(sim_rep, dtype=torch.float32)
        out, (h_n, h_c) = self.rnn(sim_rep, None)
        out = (out[:, -1, :])
        out = self.hidden(out)
        out = self.out(out)
        return out

    def forward_cls(self, x):
        c = torch.split(x, 1, dim = 0)
        x_l = c[0]
        x_r = c[1]

        sim_rep = x_l-x_r
        #sim_rep = torch.tensor(sim_rep, dtype=torch.float32)
        out, (h_n, h_c) = self.rnn(sim_rep, None)
        out = (out[:, -1, :])
        out = self.hidden(out)
        out = self.out_cls(out)
        return out
    @staticmethod
    def euclidean_distance(l, r):
        dis = torch.sub(l, r)
        dis = dis.pow(2)
        return dis
        
    @staticmethod
    def cos_distance(l,r):
        l = F.normalize(l, dim=-1)
        r = F.normalize(r, dim=-1)
        cose = torch.mm(l,r)
        return 1 - cose
    @staticmethod
    def element_wise(l,r):            
        pass


class DMFormatDataset(data.Dataset):
    def __init__(self, train_pt, eb_model, embeding_style, schema):
        self.data  = pd.read_csv(train_pt)
       # self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.train_label = self.data['label']
        self.label =  torch.tensor(np.array(self.data["label"].values), dtype=torch.long)
        self.label = torch.reshape(self.label, (-1,1))
        self.eb_model = eb_model
        self.schema = schema
        self.length = len(self.label)
        train_eb_ = []
        for index in range(self.length):
            eb_l = []
            eb_r = []
            for attr in self.schema:
                attr_l = "left_"+attr
                attr_r = "right_"+attr
                tuple_ = self.data.loc[index]
                t_l = tuple_[attr_l]
                t_r = tuple_[attr_r]
                eb_l.append(self.eb_model.avg_embeding(str(t_l)))
                eb_r.append(self.eb_model.avg_embeding(str(t_r)))
            eb_l = np.reshape(eb_l, (-1, 4, 100))
            eb_r = np.reshape(eb_r, (-1, 4, 100))
            label = self.label[index]
            eb_ = np.concatenate((eb_l, eb_r), axis=0)
            train_eb_.append(eb_)
            #data_ = torch.tensor(data_)
        self.train_eb = torch.tensor(train_eb_, dtype=torch.float32)

    def __getitem__(self, index):
        data_ = self.train_eb[index]
        label_ = self.label[index]
        return data_, label_

    def __len__(self):
        return len(self.label)

class CLearnClearnERDataset(data.Dataset):
    def __init__(self, attr_name, train_pth, bin_pth, embeding_style):
        self.train_pth = train_pth
        self.data  = pd.read_csv(self.train_pth)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data_size = len(self.data)
        self.train_label = self.data['label']
        self.label =  torch.tensor(np.array(self.data["label"].values))
        self.label = self.label.view(-1,1)
        #self.attr_pair = self.get_attr(attr_name=attr_name)
        self.eb_model = eb.FastTextEmbeding(bin_pth=bin_pth)
        self.embeding_data = self.eb_model.dataset_embeding(self.attr_pair, embeding_style)

   # def get_attr(self, attr_name):
   #     l_attr_name = "left_"+attr_name
   #     r_attr_name = "right_"+attr_name
   #     attr_pair = [self.data[l_attr_name].values, self.data[r_attr_name].values]
   #     attr_pair = np.array(attr_pair)
   #     return attr_pair.T
    def get_label(self):
        return self.label
    def __getitem__(self, index):
        label = self.label[index]
        data = self.embeding_data[index]
 #       print(self.attr_pair[index])
        data[0] = torch.tensor([data[0]])
        data[1] = torch.tensor([data[1]])
        return data, label
    def init(self):
        self.embeding_data = self.eb_model.dataset_embeding(self.attr_pair, embeding_style)

    def __len__(self):
        return len(self.label)
import numpy as np 
import pandas as pd
from functools import reduce
import os
import sys
sys.path.append("..") 
from distributed_rep import embeding

class LSH():
    class HashTable:
        def __init__(self):

            self.num_ball = 0
            self.num_bucket = 0  
            self.table = {}

    def __init__(self, num_hash_func, num_hash_table, data_l, data_r, schema, embeding_model):
        np.random.seed(1111131)
        self.schema = schema
        self.num_hash_func = num_hash_func
        self.num_hash_table = num_hash_table
        #self.hash_table = [{} for _ in range(num_hash_table)]
        self.tables = [self.make_table() for _ in range(num_hash_table)]

        self.data_l  = data_l
        self.data_r  = data_r

        self.size_hash_table = 100
        self.hash_funcs = [[np.random.randint(-1,2,(self.size_hash_table, 4)) for _ in range(num_hash_func)] for _ in range (self.num_hash_table)]
        self.embeding_model = embeding_model
        self.tuples_eb = {}
        self.tau = 0.2

    def make_table(self):
        return self.HashTable()
    def get_table(self, table_id):
        return self.tables[table_id].table
    def index(self):
        for index, row in self.data_l.iterrows():
            eb = []
            for atom in self.schema:
                eb.append(np.array(self.embeding_model.get_embeding(str(row[atom]))))
            eb = np.array(eb)
            table_id = 0
            for hash_fam in self.hash_funcs:
                for h_ in hash_fam:
                    pos = np.dot(eb, h_)
                    pos = np.dot(pos, np.ones(4))
                    pos[pos < self.tau] = 0 
                    pos[pos >= self.tau] = 1 
                    pos = int(reduce(lambda a,b: 2*a+b, pos))
                    table_ =  self.get_table(table_id)
                    if(pos in table_):
                        table_[pos][row['id']] = eb
                    else:
                        table_[pos] = {row['id']: eb}
                table_id += 1
        for index, row in self.data_r.iterrows():
            eb = []
            for atom in self.schema:
                eb.append(np.array(self.embeding_model.get_embeding(str(row[atom]))))
            eb = np.array(eb)
            self.tuples_eb[row['id']] = eb
            table_id = 0
            for hash_fam in self.hash_funcs:
                for h_ in hash_fam:
                    pos = np.dot(eb, h_)
                    pos = np.dot(pos, np.ones(4))
                    pos[pos < self.tau] = 0 
                    pos[pos >= self.tau] = 1 
                    pos = int(reduce(lambda a,b: 2*a+b, pos))
                    table_ =  self.get_table(table_id)
                    if pos in table_:
                        table_[pos][row['id']] = eb
                    else:
                        table_[pos] = {row['id']: eb}
                table_id += 1
        return

    def show_hash_table(self):
        print("*****************")
        for table_id in range(self.num_hash_table):
            print("####### table: ", table_id, "#########")
            table_ =  self.get_table(table_id)
            for bucket_id in table_:
                print("bucket_id", bucket_id, " size: ", len(table_[bucket_id]))
               # for key in table_[bucket_id]:
               #     print(key)

        print("*****************")


if __name__ == '__main__':
    schema = ['title', 'authors', 'venue', 'year']
    embeding_model = embeding.FastTextEmbeding(load_model='bin', source_pt  = '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin')
    eb = embeding_model.get_embeding('applying job')
    lsh = LSH(2, 2, '/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/DBLP2.csv', '/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/ACM.csv', schema, embeding_model)
    lsh.index()
    lsh.show_hash_table()
    #print(lsh.hash_table)
    #distributed_rep.embeding.FastTextEmbeding()


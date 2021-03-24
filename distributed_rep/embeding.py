import numpy as np
import time
import sqlite3
import numpy as np
import io
from abc import ABC, abstractmethod
import fasttext

class EmbedingModel():
    def __init__(self):
        pass
    
    @abstractmethod
    def get_embeding(self, str_):
        pass

class GloveEmbeding(EmbedingModel):
    def __init__(self, load_model=None, source_pt=None):
        if(load_model == 'db'):
            print("load", source_pt)
            sqlite3.register_adapter(np.ndarray, self.adapt_array)
            sqlite3.register_converter("array", self.convert_array)
            self.embeding_sql_cur = sqlite3.connect(source_pt, detect_types=sqlite3.PARSE_DECLTYPES).cursor()
        else:
            self.glove_pt = source_pt
            self.glove = open(self.glove_pt, 'r')
            sqlite3.register_adapter(np.ndarray, self.adapt_array)
            sqlite3.register_converter("array", self.convert_array)
        
            self.embeding_sql_cur = self.get_embeding_sql()

    def get_embeding_sql(self):
        con = sqlite3.connect('glove.42B.300d.db', detect_types=sqlite3.PARSE_DECLTYPES) 
        cur = con.cursor()
        cur.execute('''CREATE TABLE word_embeding
            (word CHAR(50),
            word_vec array);''')

        glove = open(self.glove_pt, 'r')
        word = list()
        word_vector = list()
        line = glove.readline()  # 一行一行的读取，返回str

        count = 0
        while line:
            line = list(line.split())
            count+=1
            print(count, line[0])
            word.append(line[0])
            word_vector.append(line[1:])

            vec = np.array(list(map(float,line[1:])))
            vec = np.reshape(vec, [-1])
            cur.execute("insert into word_embeding (word, word_vec) values (?, ?)", (line[0], vec))
            line = glove.readline()
            
        con.commit()
        return cur

    @staticmethod
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def get_embeding(self, str_):
        sql = "select * from word_embeding where word = \"{}\"".format(str_)
        cursor = self.embeding_sql_cur.execute(sql)
        for row in cursor:
            #print(row)
            return row[1]
            break

class FastTextEmbeding(EmbedingModel):
    def __init__(self, load_model, source_pt):
        self.load_model = load_model
        print("load", source_pt)
        self.model = fasttext.load_model(source_pt)

    def seq_embeding(self, seq_words):
        if(type(seq_words) == float or seq_words is None):
            seq_words = "?"
        words = seq_words.split()
        vec_seq = []
        for w_ in words:
            vec_ = self.model.get_word_vector(w_)   
            vec_seq.append(vec_)
        return np.array(vec_seq)

    def avg_embeding(self, seq_words):
        vec_seq = self.seq_embeding(seq_words)
        vec_seq = np.array(vec_seq)
    
        len_ = len(vec_seq)
        sum_vec = np.zeros(np.shape(vec_seq)[1])
        for vec_ in vec_seq:
            sum_vec += vec_
        avg_vec =  sum_vec / len_
        return np.array(avg_vec)

    def sum_embeding(self, seq_words):
        vec_seq = self.avg_embeding(seq_words)
        vec_seq = np.array(vec_seq)
        len_ = len(vec_seq)
        sum_vec = np.zeros(np.shape(vec_seq)[1])
        for vec_ in vec_seq:
            sum_vec += vec_
        return np.array(sum_vec)

    def dataset_embeding(self, data_, embeding_style):
        data_embeding = []
        switch = {
            'avg':self.avg_embeding, 
            'max':self.sum_embeding,
            'seq':self.seq_embeding,
        }
        eb_model = switch.get(embeding_style)
        for i in data_:
           eb_l = eb_model(i[0])
           eb_r = eb_model(i[1])
           data_embeding.append([eb_l, eb_r])
        return np.array(data_embeding)
    def get_embeding(self, str_):
        return self.avg_embeding(str_)
    
    
if __name__ == '__main__':
    #embeding_model = GloveEmbeding(load_model = 'db', source_pt = 'glove.42B.300d.db')
    #eb = embeding_model.get_embeding('big')

    EmbedingModel = FastTextEmbeding(load_model='bin', source_pt  = '/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm.bin')
    eb = EmbedingModel.get_embeding('applying job')
    print(eb)
    t0 = time.time()
    print (time.time() - t0, " sec.")
import os
from tracemalloc import start
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, gen_path, multiplier, hidden_size, gen_layers, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, augment=None):
        
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.augment = augment
        self.multiplier = 1
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        # replace the training set with the generated data
        self.original = False
        if flag == 'train':
            self.syn_x = []
            self.syn_y = []
            if '.npy' in gen_path:
                self._replace_train_with_npy(gen_path, multiplier)
            elif '' == gen_path:
                self.original = True

    def __read_data__(self):
        self.scaler = StandardScaler()
        if '.csv' in self.data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        else:
            df_raw = np.loadtxt(os.path.join(self.root_path, self.data_path), delimiter=',')
            df_raw = pd.DataFrame(df_raw)
            _columns = list(df_raw.columns)
            df_raw['date'] = None
            df_raw = df_raw[['date'] + _columns]
        if self.data_path in ['ETTh1.csv', 'ETTh2.csv']:
            border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
            border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        elif self.data_path == 'ILI.csv':
            border1s = [0, 580 - self.seq_len, 773 - self.seq_len]
            border2s = [580, 773, 965]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        elif self.data_path == 'us_births.csv':
            border1s = [0, 4383 - self.seq_len, 5844 - self.seq_len]
            border2s = [4383, 5844, 7304]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_stamp = np.zeros((df_data.shape[0], 1))
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        if self.original or self.set_type != 0:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len 
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]  # 0 - 24
            seq_y = self.data_y[r_begin:r_end] # 0 - 48
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.set_type == 0 and not self.original:
            seq_x = self.syn_x[index]
            seq_y = self.syn_y[index]
            seq_x_mark = np.array([1])
            seq_y_mark = np.array([1])

        if self.set_type == 0:
            if self.augment == 'jitter':
                seq_x = seq_x + np.random.normal(loc=0., scale=0.05, size=seq_x.shape)
            elif self.augment == 'scaling':
                factor = np.random.normal(loc=1., scale=0.1, size=(1, seq_x.shape[1]))
                seq_x = np.multiply(seq_x, factor)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        if self.original or self.set_type !=0:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        else:
            return len(self.data_x) * self.multiplier

    def _minmaxscaler(self, data):
        min_val = np.min(data, axis=0)
        data = data - min_val

        max_val = np.max(data, axis=0)
        data = data / (max_val + 1e-7)

        return min_val, max_val

    def _replace_train_with_npy(self, gen_path, multiplier):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if multiplier == '':
            multiplier = 1
        else:
            multiplier = int(multiplier)
        self.multiplier = multiplier
        generated_data = np.load(gen_path)[:length * multiplier, :, :]
        generated_data = self.scaler.transform(generated_data)
       
        self.data_x = np.zeros((length, 1))
        self.data_y = np.zeros((length, 1))
        self.data_stamp = np.zeros((length, 1))
        self.syn_x = generated_data[:, :self.seq_len, :]        
        self.syn_y = generated_data[:, self.seq_len:self.seq_len+self.pred_len, :]        
           
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, gen_path, multiplier, hidden_size, gen_layers, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None, augment=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.augment = augment
        self.multiplier = 1
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        
        # 替换掉训练集数据
        self.original = False
        if flag == 'train':
            self.syn_x = []
            self.syn_y = []
            if '.npy' in gen_path:
                self._replace_train_with_npy(gen_path, multiplier)
            elif '' == gen_path:
                self.original = True

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        if self.original or self.set_type != 0:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.set_type == 0 and not self.original:
            seq_x = self.syn_x[index]
            seq_y = self.syn_y[index]
            seq_x_mark = np.array([1])
            seq_y_mark = np.array([1])
        
        if self.set_type == 0:
            # 只有训练集才做augmentation
            if self.augment == 'jitter':
                seq_x = seq_x + np.random.normal(loc=0., scale=0.05, size=seq_x.shape)
            elif self.augment == 'scaling':
                factor = np.random.normal(loc=1., scale=0.1, size=(1, seq_x.shape[1]))
                seq_x = np.multiply(seq_x, factor)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        if self.original or self.set_type !=0:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        else:
            return len(self.data_x) * self.multiplier

    def _minmaxscaler(self, data):
        min_val = np.min(data, axis=0)
        data = data - min_val

        max_val = np.max(data, axis=0)
        data = data / (max_val + 1e-7)

        return min_val, max_val

    def _replace_train_with_timegan(self):
        pass

    def _replace_train_with_npy(self, gen_path, multiplier):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if multiplier == '':
            multiplier = 1
        else:
            multiplier = int(multiplier)
        self.multiplier = multiplier
        generated_data = np.load(gen_path)[:length * multiplier, :, :]
        generated_data = self.scaler.transform(generated_data)
       
        self.data_x = np.zeros((length, 1))
        self.data_y = np.zeros((length, 1))
        self.data_stamp = np.zeros((length, 1))
        self.syn_x = generated_data[:, :self.seq_len, :]        
        self.syn_y = generated_data[:, self.seq_len:self.seq_len+self.pred_len, :]        


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



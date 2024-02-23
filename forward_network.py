import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset,DataLoader
import os
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.pyplot import MultipleLocator
from sklearn import preprocessing
import sys
import pickle

def load_pkl(path):
    with open(path,'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
def save_pkl(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def get_mag_data(label_type):
    label_path = 'data/{}/'.format(label_type)
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').readlines()
        label_sample = []
        for i in tmp:
            label = float(i.strip().split()[-1])
            if label<0.0 or label>1.0:
                break
            else:
                label_sample.append(label)
        if len(label_sample) == len(tmp):
            label_data.append(label_sample)
            label_index.append(file.split('.')[0].split('magnitude')[1])
    label_data = np.array(label_data)

    data_path = 'data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    return data, label_data,label_index

def get_phaseX_data():
    label_path = './data/phaseX/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').readlines()
        label_sample = []
        for i in tmp:
            label = float(i.strip().split()[-1])
            if label<-180.0 or label>180.0:
                break
            else:
                label_sample.append(label)
        if len(label_sample) == len(tmp):
            label_data.append(label_sample)
            label_index.append(file.split('.')[0].split('phase')[1])
    label_data = np.array(label_data)/180

    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([[float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    return data, label_data,label_index

def get_unphaseX_data():
    label_path = './data/unphaseX/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').readlines()
        label_sample = [float(i.strip().split()[-1]) for i in tmp]
        label_data.append(label_sample)
        label_index.append(file.split('.')[0].split('unphase')[1])
    label_data = np.array(label_data)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    label_data = min_max_scaler.fit_transform(label_data)
    save_pkl(min_max_scaler,'unphaseX_mm.pkl')
    
    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([[float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    return data, label_data, label_index

def get_ER_data():
    label_path = './data/magnitudeERlist/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').read().strip().split()[:-1]
        label_data.append([float(i) for i in tmp])
        label_index.append(file.split('.')[0].split('magnitude')[1])
    label_data = np.array(label_data)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    label_data = min_max_scaler.fit_transform(label_data)
    save_pkl(min_max_scaler,'ER_mm.pkl')
    
    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    return data, label_data, label_index

def get_PCR_data():
    label_path = './data/magnitudePCRlist/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').read().strip().split()[:-1]
        label_data.append([float(i) for i in tmp])
        label_index.append(file.split('.')[0].split('magnitude')[1])
    label_data = np.array(label_data)/1.6

    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    return data, label_data,label_index

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.cnn1 = nn.Conv1d(
                        in_channels=25,
                        out_channels=hidden_size,
                        kernel_size = cnn_kernel_size
                        )
        
        self.flatten = nn.Flatten()
        
        
        self.cnn2 = nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size = cnn_kernel_size
                        )
        self.max_pooling = nn.MaxPool1d(kernel_size = (25-cnn_kernel_size+1))
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size , classes_num)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()


    def forward(self,x):
        hidden = x.permute(0,2,1)
        hidden = self.cnn1(hidden)
        hidden = self.max_pooling(hidden)
        hidden= self.bn1(hidden)
        hidden = torch.squeeze(hidden,-1)
        hidden=self.relu(hidden)
        
        hidden=self.dp(hidden)

        y=self.fc3(hidden)

        return y
    
class LSTM_Net(nn.Module):
    def __init__(self):
        super(LSTM_Net, self).__init__()

        self.lstm = nn.LSTM(input_size=25,hidden_size=hidden_size,num_layers=1,
                        bias=True,batch_first=True,dropout=drop_rate,bidirectional=True)
        self.fc3 = nn.Linear(hidden_size*2 , classes_num)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()
    def forward(self,x):
        
        hidden=self.lstm(x)[0][:,-1,:]
        hidden=self.relu(hidden)
        hidden=self.dp(hidden)

        y=self.fc3(hidden)

        return y

class GRU_Net(nn.Module):
    def __init__(self):
        super(GRU_Net, self).__init__()

        self.lstm = nn.GRU(input_size=25,hidden_size=hidden_size,num_layers=1,
                        bias=True,batch_first=True,dropout=0.3,bidirectional=True)
        self.fc3 = nn.Linear(hidden_size*2 , classes_num)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()

    def forward(self,x):
        
        hidden=self.lstm(x)[0][:,-1,:]
        hidden=self.relu(hidden)
        hidden=self.dp(hidden)

        y=self.fc3(hidden)

        return y
    
class Transformer_Net(nn.Module):
    def __init__(self):
        super(Transformer_Net, self).__init__()
        self.fc1=nn.Linear(25,hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=5)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.max_pooling = nn.MaxPool1d(kernel_size = 25)
        self.fc3 = nn.Linear(hidden_size , classes_num)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()


    def forward(self,x):
        hidden = self.fc1(x)
        hidden = self.transformer(hidden)
        hidden = hidden.permute(0,2,1)
        hidden = self.max_pooling(hidden)
        hidden = torch.squeeze(hidden,-1)
        hidden=self.relu(hidden)
        hidden=self.dp(hidden)

        y=self.fc3(hidden)
        return y
    
    
def train(model,model_name, train_dataloader,criterion,optimizer,epochs,device):
    model=model.train()
    for epoch in range(epochs):
        for x,y in train_dataloader:
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = criterion(y_.float(),y.float())
            loss.backward()
            optimizer.step() 
            
def get_MSE(model,model_name,dataloader):
    model=model.eval()
    y_true=[]
    y_pred=[]
    su = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_ = model(x)
            y_ = y_.cpu().detach().numpy()

            y_true.extend(y.cpu().detach().numpy())
            y_pred.extend(y_)
    return mean_squared_error(y_pred,y_true)    
    
if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    label_path = sys.argv[1]
    if label_path == '0':
        label_path = 'magnitudeX'
        X,y,label_index = get_mag_data(label_path)
    elif label_path == '1':
        label_path = 'magnitudeY'
        X,y,label_index = get_mag_data(label_path)
    elif label_path == '2':
        label_path = 'phaseX'
        X,y,label_index = get_phaseX_data()
    elif label_path == '3':
        label_path = 'unphaseX'
        X,y,label_index,data_min,data_max = get_unphaseX_data()   
        save_pkl(data_max,'unphaseX_data_max.pkl')
        save_pkl(data_min,'unphaseX_data_min.pkl')
    elif label_path == '4':
        label_path = 'magnitudeERlist'
        X,y,label_index,data_min,data_max = get_ER_data()   
        save_pkl(data_max,'magnitudeERlist_data_max.pkl')
        save_pkl(data_min,'magnitudeERlist_data_min.pkl')
    elif label_path == '5':
        label_path = 'magnitudePCRlist'
        X,y,label_index = get_PCR_data()   
    torch.backends.cudnn.deterministic=True

    classes_num = y.shape[-1]
    epoch = 50
    batch = 32
    cnn_kernel_size = 5
    drop_rate = 0.3
    kf = KFold(n_splits=5,shuffle=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    for model_name in ['CNN','LSTM','GRU','Transformer']:
        result_path = './{}_result/{}_results.txt'.format(label_path, model_name)
        result_file = open(result_path,'w',encoding='utf8')
        best_test_mse = 1.0
        for hidden_size in range(10,130,10):
            all_test_mse = 0.0
            print('{} {} test_mse\n'.format(model_name, hidden_size))
            result_file.write('{} {} test_mse\n'.format(model_name,hidden_size))
            for idx,(train_index,test_index) in enumerate(kf.split(X)):
                model = eval('{}_Net()'.format(model_name))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)

                train_x,train_y = X[train_index],y[train_index]
                test_x,test_y = X[test_index],y[test_index]

                train_x = torch.from_numpy(train_x).float()
                train_y = torch.from_numpy(train_y)

                train_dataset = TensorDataset(train_x,train_y)
                train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch)


                test_x = torch.from_numpy(test_x).float()
                test_y = torch.from_numpy(test_y)
                test_dataset = TensorDataset(test_x,test_y)
                test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=batch)

                train(model,model_name,train_dataloader,criterion,optimizer,epoch,device)

                test_mse = get_MSE(model,model_name,test_dataloader)

                all_test_mse += test_mse
                print('{} {}\n'.format(idx+1,test_mse))
                result_file.write('{} {}\n'.format(idx+1,test_mse))

            torch.save(model,'./{}_result/{}_model_{}.pt'.format(label_path, model_name, hidden_size))
            print('avg_mse {}\n'.format(all_test_mse/5))
            result_file.write('avg_mse{}\n'.format(all_test_mse/5))
            if all_test_mse/5 < best_test_mse:
                best_test_mse = all_test_mse/5
        print('best_mse {}\n'.format(best_test_mse)) 
        result_file.write('best_mse {}\n'.format(best_test_mse))   
        result_file.close()
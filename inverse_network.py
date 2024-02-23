import torch
from torch import nn
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset,DataLoader
import os
from sklearn.metrics import *
from sklearn import preprocessing

def get_magnitudeX_data():
    magX_label_path = './data/{}/'.format('magnitudeX')
    magX_label_files = [i.split('.')[0].split('magnitude')[1] for i in os.listdir(magX_label_path)]
    magX_label_data = []
    magX_label_index = []
    for file in magX_label_files:
        tmp = open(magX_label_path+'magnitude{}.txt'.format(file),'r',encoding='utf8').readlines()
        label_sample = []
        for i in tmp:
            label = [float(i) for i in i.strip().split()]
            label[0] = label[0]/2.0
            if label[-1]<0.0 or label[-1]>1.0:
                break
            else:
                label_sample.append(label)
        if len(label_sample) == len(tmp):
            magX_label_data.append(label_sample)
            magX_label_index.append(file)
    magX_label_data = np.array(magX_label_data)

    data_path = 'data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in magX_label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    data = data.reshape(data.shape[0],625)
    return data, magX_label_data, data_index

def get_single_magnitudeX_data():
    magX_label_path = './data/{}/'.format('magnitudeX')
    magX_label_files = [i.split('.')[0].split('magnitude')[1] for i in os.listdir(magX_label_path)]
    magX_label_data = []
    magX_label_index = []
    for file in magX_label_files:
        tmp = open(magX_label_path+'magnitude{}.txt'.format(file),'r',encoding='utf8').readlines()
        label_sample = []
        for i in tmp:
            label = float(i.strip().split()[-1])
            if label<0.0 or label>1.0:
                break
            else:
                label_sample.append([label])
        if len(label_sample) == len(tmp):
            magX_label_data.append(label_sample)
            magX_label_index.append(file)
    magX_label_data = np.array(magX_label_data)

    data_path = 'data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in magX_label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    data = data.reshape(data.shape[0],625)
    return data, magX_label_data, data_index

def get_phaseX_data():
    label_path = './data/phaseX/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').readlines()
        label_sample = []
        for i in tmp:
            label = [float(i) for i in i.strip().split()]
            label[0] = label[0]/2.0
            label[-1] = label[-1]/180.0
            if label[-1]<-1.0 or label[-1]>1.0:
                break
            else:
                label_sample.append(label)
        if len(label_sample) == len(tmp):
            label_data.append(label_sample)
            label_index.append(file.split('.')[0].split('phase')[1])
    label_data = np.array(label_data)
    
    data_path = 'data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    data = data.reshape(data.shape[0],625)
    return data, label_data, data_index

def get_unphaseX_data():
    label_path = './data/unphaseX/'
    label_files = os.listdir(label_path)
    label_data = []
    label_index = []
    for file in label_files:
        tmp = open(label_path+file,'r',encoding='utf8').readlines()
        label_sample = []
        label_left = []
        for i in tmp:
            label = [float(i) for i in i.strip().split()]
            label[0] = label[0]/2.0
            label_left.append(label[0])
            label_sample.append(label[-1])
        label_index.append(file.split('.')[0].split('unphase')[1])
        label_data.append(label_sample)
    label_data = np.array(label_data)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    label_data = min_max_scaler.fit_transform(label_data)
    data_max = min_max_scaler.data_max_
    data_min = min_max_scaler.data_min_
    label_data = np.concatenate((np.expand_dims(np.array([label_left]*len(label_files)),-1), np.expand_dims(label_data,-1)),axis=-1)
    
    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]

    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([[float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    data = data.reshape(data.shape[0],625)
    return data, label_data, data_index

def get_magnitudeERlist_data():
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
    data_max = min_max_scaler.data_max_
    data_min = min_max_scaler.data_min_
    
    data_path = './data/matrix/'
    data_index = ['matrix{}.txt'.format(i) for i in label_index]
    data = []
    for file in data_index:
        tmp = open(data_path+file,'r',encoding='utf8').readlines()
        data.append([ [float(j) for j in i.strip().split()] for i in tmp])
    data = np.array(data)
    data = data.reshape(data.shape[0],625)
    return data, np.expand_dims(label_data,-1), data_index

def get_magnitudePCRlist_data():
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
    data = data.reshape(data.shape[0],625)
    return data, np.expand_dims(label_data,-1), data_index

class MLP_Net(nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()
        self.fc1=nn.Linear(sentence_len,hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc2=nn.Linear(hidden_size,classes_num)
        self.fc3 = nn.Linear(classes_num , classes_num)
        self.dp=nn.Dropout(0.3)
        self.relu=nn.ReLU()


    def forward(self,x):
        hidden = torch.squeeze(x,-1)
        hidden=self.fc1(hidden)
        hidden= self.bn1(hidden)
        hidden= self.fc2(hidden)
        hidden=self.relu(hidden)
        hidden=self.dp(hidden)

        y=self.fc3(hidden)

        return y
    
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.cnn1 = nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size = cnn_kernel_size
                        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.max_pooling = nn.MaxPool1d(kernel_size = (sentence_len-cnn_kernel_size+1))
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size , classes_num)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()


    def forward(self,x):
        hidden = self.fc1(x)
        hidden = hidden.permute(0,2,1)
        hidden = self.cnn1(hidden)
        hidden = self.max_pooling(hidden)
        hidden = torch.squeeze(hidden,-1)
        hidden = self.relu(hidden)
        hidden = self.bn1(hidden)
        hidden = self.dp(hidden)

        y = self.fc3(hidden)

        return y

class LSTM_Net(nn.Module):
    def __init__(self):
        super(LSTM_Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,
                        bias=True,batch_first=True,dropout=drop_rate,bidirectional=True)
        self.fc3 = nn.Linear(hidden_size*2 , classes_num)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size*2)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()




    def forward(self,x):
        hidden = self.fc1(x)
        hidden = self.lstm(hidden)[0][:,-1,:]
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dp(hidden)

        y = self.fc3(hidden)

        return y

class GRU_Net(nn.Module):
    def __init__(self):
        super(GRU_Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.lstm = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,
                        bias=True,batch_first=True,dropout=0.3,bidirectional=True)
        self.fc3 = nn.Linear(hidden_size*2 , classes_num)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size*2)
        self.dp=nn.Dropout(drop_rate)
        self.relu=nn.ReLU()




    def forward(self,x):
        hidden = self.fc1(x)
        hidden = self.lstm(hidden)[0][:,-1,:]
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dp(hidden)

        y = self.fc3(hidden)

        return y

class Transformer_Net(nn.Module):
    def __init__(self):
        super(Transformer_Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=5)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.max_pooling = nn.MaxPool1d(kernel_size = sentence_len)
        self.bn1 = nn.BatchNorm1d(num_features = hidden_size)
        self.fc3 = nn.Linear(hidden_size , classes_num)
        self.dp = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self,x):
        hidden = self.fc1(x)
        hidden = self.transformer(hidden)
        hidden = hidden.permute(0,2,1)
        hidden = self.max_pooling(hidden)
        hidden = torch.squeeze(hidden,-1)
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dp(hidden)

        y = self.sigmoid(self.fc3(hidden))

        return y
class num_one_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        discrete_result = torch.where(x>=0.5, 1.0, 0.0)
        num_one = torch.sum(discrete_result,-1)
        n_loss = torch.abs(torch.tensor(312).to(device)-num_one)*0.001
        return torch.mean(n_loss)
    
def train(model,train_dataloader,mse_criterion,one_criterion,optimizer,epochs,device):
    model=model.train()
    for epoch in range(epochs):
        for x,y in train_dataloader:
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()

            y_pred=model(x)

            mse_loss = mse_criterion(y_pred.float(),y.float())
            one_loss = one_criterion(y_pred.float())
            loss = (mse_loss+one_loss)/2
            loss.backward()
            optimizer.step()

def get_MSE(model,dataloader):
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
    label_type = sys.argv[1]
    torch.backends.cudnn.deterministic=True
    epoch = 50
    batch = 32
    cnn_kernel_size = 5
    drop_rate = 0.3
    kf = KFold(n_splits=5,shuffle=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    if label_type == 'magnitudeX':
        y,X,label_index = get_magnitudeX_data()
    elif label_type == 'magnitudePCRlist':
        y,X,label_index = get_magnitudePCRlist_data()
        cnn_kernel_size = 2
    elif label_type == 'magnitudeERlist':
        y,X,label_index = get_magnitudeERlist_data() 
        cnn_kernel_size = 2
    elif label_type == 'phaseX':
        y,X,label_index = get_phaseX_data()
    elif label_type == 'unphaseX':
        y,X,label_index = get_unphaseX_data()

    input_size = X.shape[-1]
    sentence_len = X.shape[1]
    classes_num = y.shape[-1]
    model_list = ['CNN','LSTM','GRU','Transformer']#'CNN','LSTM','GRU','Transformer'
    for model_name in model_list:
        result_path = './reverse_{}_result/{}_results.txt'.format(label_type, model_name)
        result_file = open(result_path,'w',encoding='utf8')
        min_mse = 1
        for h in list(range(10,150,10)):
            all_test_mse = 0.0
            hidden_size = h
            print('model:{}, hidden_size:{}'.format(model_name, hidden_size))

            for idx,(train_index,test_index) in enumerate(kf.split(X)):
                model = eval('{}_Net()'.format(model_name))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)

                print('第{}折交叉'.format(idx+1))
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


                train(model,train_dataloader,criterion,optimizer,epoch,device)

                test_mse = get_MSE(model,test_dataloader)
                print('test_mse: {}'.format(test_mse))

                all_test_mse += test_mse
                result_file.write('{} {}\n'.format(idx+1,test_mse))
            torch.save(model,'./{}_result/{}_model_{}.pt'.format(label_path, model_name, hidden_size))
            print('5 fold result: {}'.format(all_test_mse/5))
            result_file.write('{} {} \n'.format(idx+1,all_test_mse/5))
        result_file.close()


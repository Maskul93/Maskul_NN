##Import libraries
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import time
import datetime
import os
import sys
import csv
from torch import backends
from beautifultable import BeautifulTable
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#To be overwirtten by clients
_nmuscles = 10
_spw = 20
_batch_size = 32

#List of all models. Common activation function: ReLu. Common dp_ratio=0.5. Last activation function: sigmoid.

#1 hidden layer
class Model0(torch.nn.Module):
    def __init__(self):
        super(Model0,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,32)
        self.l2 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        y_pred=self.sigmoid(self.l2(out))
        return y_pred


#2 hidden layers
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,64)
        self.l2 = torch.nn.Linear(64,32)
        self.l3 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        y_pred = self.sigmoid(self.l3(out))
        return y_pred

#2 hidden layers w/dropout
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,64)
        self.l1_dropout = nn.Dropout(p = 0.5)
        self.l2 = torch.nn.Linear(64,32)
        self.l2_dropout = nn.Dropout(p = 0.5)
        self.l3 = torch.nn.Linear(32,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = F.relu(self.l1_dropout(self.l1(x)))
        out = F.relu(self.l2_dropout(self.l2(out)))
        y_pred = self.sigmoid(self.l3(out))
        return y_pred

#3 hidden layers
class Model3(torch.nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, 128)
        self.l4 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        y_pred = self.sigmoid(self.l4(out))
        return y_pred

#3 hidden layers w/dropout
class Model4(torch.nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l2_dropout = nn.Dropout(p=0.5)
        self.l3 = torch.nn.Linear(512, 128)
        self.l3_dropout = nn.Dropout(p=0.5)
        self.l4 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2_dropout(self.l2(out)))
        out = F.relu(self.l3_dropout(self.l3(out)))
        y_pred = self.sigmoid(self.l4(out))
        return y_pred

#1D convnet w/o dropout
class Model5(nn.Module):
    def __init__(self):
        super(Model5,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5*_nmuscles),stride=_nmuscles)
        self.mp = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(8,128)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        #print(x.shape)
        x = F.relu(self.mp(self.conv1(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        y_pred = self.fc3(x)
        return F.sigmoid(y_pred)

#1D Convnet -> Stride=_nmuscles. Multi Filter:2x_nmuscles,4x,6x,8x. Multi channel:20.
class Model6(nn.Module):
    def __init__(self,**kwargs):
        super(Model6,self).__init__()
        self.FILTERS=[5*_nmuscles,10*_nmuscles]
        conv_out_channels = 20 
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(260,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,1)

    def test_dim(self, x):
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")
        x = torch.cat(x,2)
        x= x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))
        x = F.relu(self.fc1(x))
        print("Out: " + str(x.shape))
        x = F.relu(self.fc2(x))
        print("Out: " + str(x.shape))
        x = F.relu(self.fc3(x))
        print("Out: " + str(x.shape))
        x = F.sigmoid(self.fc4(x))
        print("Out: " + str(x.shape))
        
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        #print("INPUT SIZE: " + str(x.shape))
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x= x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))

class Model7(nn.Module):
    def __init__(self,**kwargs):
        super(Model7,self).__init__()
        self.FILTERS=[3*_nmuscles,5*_nmuscles,10*_nmuscles]
        conv_out_channels = 10
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(160,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 3")
        x = self.fc3(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 4")
        x = self.fc4(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        #print("INPUT SIZE: " + str(x.shape))
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x= x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))
    
#1D Convnet -> Stride=_nmuscles.
class Model8(nn.Module):
    def __init__(self,**kwargs):
        super(Model8,self).__init__()
        self.FILTERS=[5*_nmuscles,10*_nmuscles]
        conv_out_channels = 10 
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        self.FILTERS_2=[3,5]
        self.convs_2 = nn.ModuleList([nn.Conv1d(in_channels=10, out_channels=conv_out_channels*2, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS_2])
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(80,32)
        self.fc2 = nn.Linear(32,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do convnets")
        x = [F.relu(conv(i)) for i in x for conv in self.convs_2]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = [F.relu(conv(i)) for i in x for conv in self.convs_2]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))

#1D Convnet -> Stride=_nmuscles.
class Model9(nn.Module):
    def __init__(self,**kwargs):
        super(Model9,self).__init__()
        self.FILTERS=[5*_nmuscles,10*_nmuscles]
        conv_out_channels = 10
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(130,32)
        self.fc2 = nn.Linear(32,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))

#1D Convnet -> Stride=_nmuscles.
class Model10(nn.Module):
    def __init__(self,**kwargs):
        super(Model10,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles,10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.MaxPool1d(2)
        self.FILTERS_2=[3,5]
        self.convs_2 = nn.ModuleList([nn.Conv1d(in_channels=20, out_channels=conv_out_channels*2, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS_2])
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(240,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do convnets")
        x = [F.relu(conv(i)) for i in x for conv in self.convs_2]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 3")
        x = self.fc3(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 4")
        x = self.fc4(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = [F.relu(conv(i)) for i in x for conv in self.convs_2]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))
    
#1 hidden layer
class Model11(torch.nn.Module):
    def __init__(self):
        super(Model11,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,128)
        self.l2 = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        y_pred=self.sigmoid(self.l2(out))
        return y_pred
    
    
class Model12(nn.Module):
    def __init__(self,**kwargs):
        super(Model12,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.AvgPool1d(3)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(280,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
#1 hidden layer
class Model13(torch.nn.Module):
    def __init__(self):
        super(Model13,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,256)
        self.l2 = torch.nn.Linear(256,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        y_pred=self.sigmoid(self.l2(out))
        return y_pred
    
#1 hidden layer
class Model14(torch.nn.Module):
    def __init__(self):
        super(Model14,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,256)
        self.l2 = torch.nn.Linear(256,128)
        self.l3 = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        y_pred=self.sigmoid(self.l3(out))
        return y_pred
    
class Model15(nn.Module):
    def __init__(self,**kwargs):
        super(Model15,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 40
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.AvgPool1d(5)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
    
 #1 hidden layer
class Model16(torch.nn.Module):
    def __init__(self):
        super(Model16,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        y_pred=self.sigmoid(self.l4(out))
        return y_pred
    
 #1 hidden layer
class Model17(torch.nn.Module):
    def __init__(self):
        super(Model17,self).__init__()
        self.l1 = torch.nn.Linear(_spw*_nmuscles,1024)
        self.l2 = torch.nn.Linear(1024,512)
        self.l3 = torch.nn.Linear(512,256)
        self.l4 = torch.nn.Linear(256,128)
        self.l5 = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        y_pred=self.sigmoid(self.l5(out))
        return y_pred
    
class Model18(nn.Module):
    def __init__(self,**kwargs):
        super(Model18,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 80
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.AvgPool1d(5)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(640,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
class Model19(nn.Module):
    def __init__(self,**kwargs):
        super(Model19,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.AvgPool1d(3)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(280,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))
    
class Model20(nn.Module):
    def __init__(self,**kwargs):
        super(Model20,self).__init__()
        self.FILTERS=[10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        #self.mp = nn.AvgPool1d(4)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(220,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        #print("Do maxpool")
        #x = [self.mp(i) for i in x]   
        #for i in x:
            #print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        #x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
class Model21(nn.Module):
    def __init__(self,**kwargs):
        super(Model22,self).__init__()
        self.FILTERS=[5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        #self.mp = nn.AvgPool1d(4)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(540,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        #print("Do maxpool")
        #x = [self.mp(i) for i in x]   
        #for i in x:
            #print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        #x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))

    
class Model22(nn.Module):
    def __init__(self,**kwargs):
        super(Model22,self).__init__()
        self.FILTERS=[3*_nmuscles, 5*_nmuscles, 10*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        #self.mp = nn.AvgPool1d(3)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(900,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        #print("Do maxpool")
        #x = [self.mp(i) for i in x]   
        #for i in x:
        #    print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        #x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
    
class Model23(nn.Module):
    def __init__(self,**kwargs):
        super(Model23,self).__init__()
        self.FILTERS=[10*_nmuscles, 20*_nmuscles]
        conv_out_channels = 20
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=k_size, stride=_nmuscles) for k_size in self.FILTERS])
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=15,stride=5)
        self.mp = nn.AvgPool1d(5)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(360,128)
        self.fc2 = nn.Linear(128,1)

    def test_dim(self,x):
        for f in self.FILTERS:
            print("Filter size: " + str(f))
        print("Input:" + str(x.shape))
        print("Features: " + str(_nmuscles))
        print("Do convnets")
        x = [F.relu(conv(x)) for conv in self.convs]
        for out in x:
            print("Out: " + str(out.shape))
        print("Do maxpool")
        x = [self.mp(i) for i in x]   
        for i in x:
            print("Out: " + str(i.shape))
        print("Do concat")    
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        print("Out: " + str(x.shape))

        print("Do Fully connected 1")
        x = self.fc1(x)
        print("Out: " + str(x.shape))
        print("Do Fully connected 2")
        x = self.fc2(x)
        print("Out: " + str(x.shape))
        
    def forward(self,x):
        x = x.view(_batch_size,1,-1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [self.mp(i) for i in x]
        x = torch.cat(x,2)
        x = x.view(32,1,-1).squeeze()
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
    
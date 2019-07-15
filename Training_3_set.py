
# coding: utf-8

# In[1]:

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


# In[2]:

##SETTINGS
doTrain = True
doEval = True
doExtractBaso = False

nfold = 17 #number of folds to train
fold_offset = 6
lr=0.01 #learning rate

batch_size = 32
val_split = 0.2 #trainset percentage allocated for devset
test_val_split = 0.1 #trainset percentage allocated for test_val set (i.e. the test set of known patients)

#cwd = os.getcwd()
#cwd = "subjects/min-max/windows_20/tr-False_sliding_1_c-False/folds_inter_no_4_24_25/"
cwd = "subjects/min-max/clean/windows_20-del_tr-True-slide-False-digits-3/folds_inter/"
subject = 1 # serve per caricare le folds da cartelle diverse
prefix_train = 'TrainFold'
prefix_test = 'TestFold'

spw=20 #samples per window
nmuscles=10 #initial number of muscles acquired

#Enable/Disable shuffle on trainset/testset
shuffle_train = False
shuffle_test= False

#Delete electrogonio signals
# 3 = Left Rectus Femuris; 5 = Left Goniometer; 9 = Right Rectus Femuris; 11 = Right Goniometer
exclude_features = True
#Only use electrogonio signals
include_only_features = False
#Features to selected/deselected for input to the networks
features_select = [9, 10] #1 to 4
#features_select = [3, 4, 7, 8, 9, 10] #1 to 4

#Select which models to run. Insert comma separated values into 'model_select' var.
#List. 0:'FF', 1:'FC2', 2:'FC2DP', 3:'FC3', 4:'FC3dp', 5:'Conv1d', 6:'MultiConv1d' 
#e.g: model_select = [0,4,6] to select FF,FC3dp,MultiConv1d

# Modelli del paper: 11 (FF2), 14 (FF4), 16 (FF5) --> Prova questi!
# FF6 per testarlo potente dopo (17)
model_lst = ['FF','FC2','FC2DP','FC3','FC3dp','Conv1d','MultiConv1d',
             'MultiConv1d_2','MultiConv1d_3', 'MultiConv1d_4', 'MultiConv1d_5', 
             'FF2', 'CNN1', 'FF3', 'FF4', 'CNN2', 'FF5', 'FF6', 'CNN3', 'CNN1-FF5', 
             'CNN1-2','CNN1-1', 'CNN1-3', 'CNN_w60', 'FF7']
model_select = [24] 

#Early stop settings
maxepoch = 100
maxpatience = 10

use_cuda = True
use_gputil = False
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:

#CUDA

if use_gputil and torch.cuda.is_available():
    import GPUtil

    # Get the first available GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=100, maxMemory=20)  # return a list of available gpus
    except:
        print('GPU not compatible with NVIDIA-SMI')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
        
    ttens = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    ttens = ttens.cuda()
    


# In[4]:

print('Is CUDA available? --> ' + str(torch.cuda.is_available()))
print('Cuda Device: ' + str(cuda_device))


# In[5]:

#Seeds
def setSeeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
setSeeds(0)


# In[6]:

#Prints header of beautifultable report for each fold
def header(model_list,nmodel,nfold,traindataset,testdataset, testdataset_U):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('MODEL: '+model_list[nmodel])
    print('Fold: '+str(nfold))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
    shape = list(traindataset.x_data.shape)
    print('Trainset fold'+str(i)+' shape: '+str(shape[0])+'x'+str((shape[1]+1)))
    shape = list(testdataset.x_data.shape)
    print('Testset (L) fold'+str(i)+' shape: '+str(shape[0])+'x'+str((shape[1]+1)))
    shape = list(testdataset_U.x_data.shape)
    print('Testset (U) fold' + str(i) + ' shape: ' + str(shape[0])+ 'x' + str((shape[1]+1)) +'\n')


# In[7]:

#Prints actual beautifultable for each fold
def table(model_list,nmodel,accuracies,precisions,recalls,f1_scores,accuracies_dev):
    table = BeautifulTable()
    table.column_headers = ["{}".format(model_list[nmodel]), "Avg", "Stdev"]
    table.append_row(["Accuracy",round(np.average(accuracies),3),round(np.std(accuracies),3)])
    table.append_row(["Precision",round(np.average(precisions),3),round(np.std(precisions),3)])
    table.append_row(["Recall",round(np.average(recalls),3),round(np.std(recalls),3)])
    table.append_row(["F1_score",round(np.average(f1_scores),3),round(np.std(f1_scores),3)])
    table.append_row(["Accuracy_dev",round(np.average(accuracies_dev),3),round(np.std(accuracies_dev),3)])    
    print(table)


# In[8]:

#Saves best model state on disk for each fold
def save_checkpoint (state, is_best, filename, logfile):
    if is_best:
        msg = "=> Saving a new best. "+'Epoch: '+str(state['epoch'])
        print (msg)
        logfile.write(msg + "\n")
        torch.save(state, filename)  
    else:
        msg = "=> Validation accuracy did not improve. "+'Epoch: '+str(state['epoch'])
        print (msg)
        logfile.write(msg + "\n")


# In[9]:

#Compute sklearn metrics: Recall, Precision, F1-score
def pre_rec (loader, model, positiveLabel):
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i,data in enumerate (loader,0):
            inputs, labels = data
            y_true = np.append(y_true,labels.cpu())
            outputs = model(inputs)
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            y_pred = np.append(y_pred,outputs.cpu())
    y_true = np.where(y_true==positiveLabel,0,1)
    y_pred = np.where(y_pred==positiveLabel,0,1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return round(precision*100,3), round(recall*100,3), round(f1_score*100,3)


# In[10]:

#Calculates model accuracy. Predicted vs Correct.
def accuracy (loader, model):
    total=0
    correct=0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data    # Carica i dati
            outputs = model(inputs)    # Tira fuori gli output
            outputs[outputs>=0.5] = 1    # Se l'output è >= 0.5 --> outputs[i] = 1
            outputs[outputs<0.5] = 0    # Se l'output è < 0.5 --> outputs[i] = 0
            total += labels.size(0)     # Calcola i labels totali
            correct += (outputs == labels).sum().item()    # Conta i corretti (quelli che matchano)
    return round((100 * correct / total),3)    # Tira fuori la percentuale di accuracy


# In[11]:

#Arrays to store metrics
accs = np.empty([nfold,1])
accs_test_val = np.empty([nfold,1])
precisions_0_U = np.empty([nfold,1])
recalls_0_U = np.empty([nfold,1])
f1_scores_0_U = np.empty([nfold,1])
precisions_1_U = np.empty([nfold,1])
recalls_1_U = np.empty([nfold,1])
f1_scores_1_U = np.empty([nfold,1])
precisions_0_L = np.empty([nfold,1])
recalls_0_L = np.empty([nfold,1])
f1_scores_0_L = np.empty([nfold,1])
precisions_1_L = np.empty([nfold,1])
recalls_1_L = np.empty([nfold,1])
f1_scores_1_L = np.empty([nfold,1])
accs_dev = np.empty([nfold,1])
times = np.empty([nfold,1])

#Calculate avg metrics on folds
def averages (vals):
    avgs = []
    for val in vals:
        avgs.append(round(np.average(val),3))
    return avgs

#Calculate std metrics on folds
def stds (vals):
    stds = []
    for val in vals:
        stds.append(round(np.std(val),3))
    return stds


# In[12]:

## -- Salva il basografico predetto su un file sfruttando l'accuracy

def save_predicted_baso(loader, model, subject_predict, model_select, num_fold):
    print("predicting baso for subject " + str(subject_predict) + " ... ")
    windows_length = 20
    predicted_baso_windowed = []
    with torch.no_grad():
        
        ## -- Estrae gli output di ciascuna window
        for i, data, in enumerate(loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            predicted_temp = outputs.tolist()
            predicted_baso_windowed.extend(predicted_temp)
        
        predicted_baso_windowed = pd.DataFrame(predicted_baso_windowed)    # Converto in DataFrame
        predicted_baso_windowed = predicted_baso_windowed.as_matrix()
        
        limit = len(predicted_baso_windowed)
        predicted_baso = np.empty([limit*windows_length,1], dtype = int)
        
        count = 0
        for j in range(0, limit):
            predicted_baso[count:count+windows_length] = predicted_baso_windowed[j]
            count += windows_length
            
        predicted_baso = pd.DataFrame(predicted_baso)
        out_path = 'subjects/min-max/clean/windows_20-del_tr-True-slide-False-digits-3/folds_inter/Report_' + str(model_select) + '/Fold_' + str(num_fold) + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #np.savetxt(out_path + 's' + str(subject_predict) + '_predicted.csv', predicted_baso, delimiter = ",")
        predicted_baso = predicted_baso.to_csv(out_path + 's' + str(subject_predict) + '_predicted.csv', index = None, header = None)


# In[13]:

## -- Salva il basografico predetto su un file sfruttando l'accuracy

def save_predicted_baso_slide(loader, model, subject_predict, model_select, num_fold):
    print("predicting baso for subject " + str(subject_predict) + " ... ")
    windows_length = 20
    predicted_baso_windowed = []
    with torch.no_grad():
        
        ## -- Estrae gli output di ciascuna window
        for i, data, in enumerate(loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            predicted_temp = outputs.tolist()
            predicted_baso_windowed.extend(predicted_temp)
        
        predicted_baso_windowed = pd.DataFrame(predicted_baso_windowed)    # Converto in DataFrame
        predicted_baso_windowed = predicted_baso_windowed.as_matrix()
        
        limit = len(predicted_baso_windowed)
        
        samples_predictions = np.empty([limit  + windows_length - 1,windows_length], dtype = int)
        for i in range(0,limit  + windows_length - 1):
            for j in range(0,windows_length):
                samples_predictions[i,j] = -1

        count = 0
        for j in range(0, limit):
            '''
            sliding_window_prediction = np.empty([windows_length,1], dtype = int)
            sliding_window_prediction[0:windows_length] = predicted_baso_windowed[j]
            indent = "wp:    "
            for i in range(0,j):
                indent += "  "
            print(indent + str(sliding_window_prediction.squeeze()))
            count += windows_length
            '''
            for i in range(0, windows_length):
                if (j+i < len(samples_predictions)):
                    samples_predictions[j+i][windows_length-i-1] = predicted_baso_windowed[j]

        #il seguente è un vettore di vettori in cui ogni elemento è composto 
        #da tutte le predizioni delle finestre che comprendono il campione. 
        # il valore -1 indica assenza di predizione, perchè i primi e gli ultimi campioni sono compresi in meno finestre
        print("\nSamples predictions:")
        print(samples_predictions)

        #Per ogni campione la predizione vincente sarà il massimo tra gli 1 e gli 0 predetti
        predicted_baso = np.empty([limit  + windows_length - 1,1], dtype = int)
        count = 0
        for sample_pred in samples_predictions:
            zeros = len(np.where(sample_pred==0)[0])
            ones = len(np.where(sample_pred==1)[0])
            if (zeros > ones):
                predicted_baso[count] = 0
            elif (zeros < ones): 
                predicted_baso[count] = 1
            else:
                predicted_baso[count] = not predicted_baso[count-1]
            #print("0:" + str(zeros), "1:" + str(ones), str(predicted_baso[count]))
            count += 1
            
        predicted_baso = pd.DataFrame(predicted_baso)
        #out_path = 'subjects/min-max/windows_20-del_tr-False-slide-False-digits-4/folds_inter/Report_' + str(model_select) + '/Fold_' + str(num_fold) + '/' 
        out_path = 'subjects/min-max/clean/windows_20-del_tr-False-slide-True-digits-3-pace-1/folds_inter/Report_' + str(model_select) + '/Fold_' + str(num_fold) + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #np.savetxt(out_path + 's' + str(subject_predict) + '_predicted.csv', predicted_baso, delimiter = ",")
        predicted_baso = predicted_baso.to_csv(out_path + 's' + str(subject_predict) + '_predicted.csv', index = None, header = None)


# In[14]:

#Shuffle
def dev_shuffle (shuffle_train,shuffle_test,val_split,traindataset,testdataset):
    train_size = len(traindataset)
    test_size = len(testdataset)
    train_indices = list(range(train_size))
    test_indices = list(range(test_size))
    split = int(np.floor(val_split * train_size))
    if shuffle_train:
        np.random.shuffle(train_indices)
    if shuffle_test:
        np.random.shuffle(test_indices) 
    train_indices, dev_indices = train_indices[split:], train_indices[:split]
    # Samplers
    tr_sampler = SubsetRandomSampler(train_indices)
    d_sampler = SubsetRandomSampler(dev_indices)
    te_sampler = SubsetRandomSampler(test_indices)
    return tr_sampler,d_sampler,te_sampler

# Questa funzione serve dividere i vari set una volta caricate le fold
# In particolare, verrà usata solo per 'dev_loader', dato che lo split 
# nei vari set è già stato fatto quando sono state create le fold
def data_split (shuffle_train,shuffle_test,val_split,test_val_split,traindataset,testdataset):
    train_size = len(traindataset)
    test_size = len(testdataset)
    train_indices = list(range(train_size))
    test_indices = list(range(test_size))
    test_val_split = int(np.floor(test_val_split * train_size)) 
    dev_split = int(np.floor(val_split * (train_size-test_val_split) + test_val_split))
    if shuffle_train:
        np.random.shuffle(train_indices)
    if shuffle_test:
        np.random.shuffle(test_indices) 
    train_indices, dev_indices, test_val_indices = train_indices[dev_split:], train_indices[test_val_split:dev_split], train_indices[:test_val_split]
    # Samplers
    tr_sampler = SubsetRandomSampler(train_indices)
    d_sampler = SubsetRandomSampler(dev_indices)
    tv_sampler = SubsetRandomSampler(test_val_indices)                
    te_sampler = SubsetRandomSampler(test_indices)
    return tr_sampler,d_sampler,tv_sampler,te_sampler

# Questa funzione serve se usiamo come input della rete 3 files:
    # TRAINFOLD
    # TESTFOLD_LEARNED
    # TESTFOLD_UNLEARNED
# L'unico file che deve essere effettivamente suddiviso, in questo caso, è il TRAINFOLD, dato che useremo
# una percentuale 'val_split' (hyperparameter) per il validation, e una percentuale 1-'val_split'
# per il vero e proprio training della rete.
# La funzione da in output i sampler da utilizzare come parametri per il DataLoader:
    # train_sampler
    # dev_sampler
def split_train ( val_split, traindataset ): 
    train_size = len(traindataset)    # lunghezza del TRAIN_DATASET
    train_indices = list(range(train_size))    # crea una lista che va da 0 -> lunghezza di train_size totale
    dev_size = int(np.floor(val_split * train_size))    # Dimensione del dev_set
    split = int(np.floor((train_size - dev_size)))    # Valore di split dei 2 set

    train_indices_bis, dev_indices = train_indices[:split], train_indices[split:]    # Indici splittati
        
    # Sampler
    tr_sampler = SubsetRandomSampler(train_indices_bis)    # Train Sampler
    d_sampler = SubsetRandomSampler(dev_indices)    # Dev Sampler
    return tr_sampler, d_sampler


# In[15]:

#Loads and appends all folds all at once
trainfolds = []    # Train set
testfolds_L = []    # Test set (LEARNED)
testfolds_U = []    # Test set (UNLEARNED)

col_select = np.array([])

#This is an hack to test smaller windows
for i in range (spw*nmuscles,200):
    col_select = np.append(col_select,i)
    
for i in range (0,spw*nmuscles,nmuscles):
    for muscle in features_select:
        col_select = np.append(col_select,muscle -1 + i)
    cols=np.arange(0,spw*nmuscles+1)

if exclude_features & (not include_only_features): #delete gonio
    for j in range(fold_offset,fold_offset + nfold):
        print("Loading fold " + str(j))
        traindata = pd.read_csv(os.path.join(cwd, prefix_train + '_' + str(j)+'.csv'),sep=',',header=None,dtype=np.float32,usecols=[i for i in cols if i not in col_select.astype(int)])
        trainfolds.append(traindata)
        testdata_L = pd.read_csv(os.path.join(cwd, prefix_test + '_L_' + str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i not in col_select.astype(int)])
        testfolds_L.append(testdata_L) 
        testdata_U = pd.read_csv(os.path.join(cwd, prefix_test + '_U_' + str(j) +'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i not in col_select.astype(int)])
        testfolds_U.append(testdata_U)
        print('Train dataset: ' + str(len(traindata)) + ' windows;')
        print('Test dataset (LEARNED): ' + str(len(testdata_L)) + ' windows;')
        print('Test dataset (UNLEARNED) ' + str(len(testdata_U)) + ' windows.')
        
elif include_only_features & (not exclude_features): #only gonio
    for j in range(fold_offset, fold_offset + nfold):
        print("Loading fold " + str(j))
        traindata = pd.read_csv(os.path.join(cwd, prefix_train + '_' + str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i in col_select.astype(int)])
        testdata_L = pd.read_csv(os.path.join(cwd, prefix_test + '_L_'+ str(j)+'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i in col_select.astype(int)])
        testdata_U = pd.read_csv(os.path.join(cwd, prefix_test + '_U_' + str(j) +'.csv'),sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i in col_select.astype(int)])
        trainfolds.append(traindata)
        testfolds_L.append(testdata_L)
        testfolds_U.append(testdata_U)
        print('Train dataset: ' + str(len(traindata)) + ' windows;')
        print('Test dataset (LEARNED): ' + str(len(testdata)) + ' windows;')
        print('Test dataset (UNLEARNED) ' + str(len(testdata_U)) + ' windows.')
        
elif (not include_only_features) & (not exclude_features): 
    for j in range(fold_offset,fold_offset + nfold):
        print("Loading fold " + str(j))
        traindata = pd.read_csv(os.path.join(cwd, prefix_train + '_' + str(j) + '.csv'),sep=',',header=None,dtype=np.float32)
        testdata = pd.read_csv(os.path.join(cwd, prefix_test + '_L_' + str(j) +'.csv'),sep=',',header=None,dtype=np.float32)
        testdata_U = pd.read_csv(os.path.join(cwd, prefix_test + '_U_' + str(j) +'.csv'),sep=',',header=None,dtype=np.float32)
        trainfolds.append(traindata)
        testfolds_L.append(testdata)
        testfolds_U.append(testdata_U)    # Aggiunto testfold UNLEARNED 
        print('Train dataset: ' + str(len(traindata)) + ' windows;')
        print('Test dataset (LEARNED): ' + str(len(testdata)) + ' windows;')
        print('Test dataset (UNLEARNED) ' + str(len(testdata_U)) + ' windows.')
else:
    raise ValueError('use_gonio and del_gonio cannot be both True')

nmuscles=int((len(traindata.columns)-1)/spw)
print('\nEach window is composed by ' + str(len(traindata.columns)) + ' samples.')
print('The number of muscles is ' + str(nmuscles))


# In[3117]:

## -- Carica il soggetto su cui vuoi predire il basografico

nmuscles = 10
subject_predict = 26
#directory_windows = '../subjects/min-max/windows_20/tr-False_sliding_1_c-False/'
#directory_windows = 'subjects/min-max/clean/windows_20-del_tr-False-slide-True-digits-3-pace-1/'
directory_windows = 'subjects/min-max/clean/windows_20-del_tr-True-slide-False-digits-3/'

subj_prefix = 's'
subj_suffix = '_norm_windows_20.csv'
file = directory_windows + subj_prefix + str(subject_predict) + subj_suffix

if doExtractBaso:
    if exclude_features & (not include_only_features): #delete gonio
        for j in range(fold_offset,fold_offset + nfold):
            testfolds_U = []

        #for i in range(spw*nmuscles,200):
        #    col_select = np.append(col_select,i)

#         for i in range (0,spw*nmuscles,nmuscles):
#             for muscle in features_select:
#                 col_select = np.append(col_select,muscle -1 + i)
#             cols=np.arange(0,spw*nmuscles+1)
        
        cols = [i for i in range(0,201)]
        
        col_select = []
        for i in range(0,200,nmuscles):
            col_select.append(8 + i)
            col_select.append(9 + i)
        
        cols = np.asarray(cols)
        col_select = np.asarray(col_select)
        
        print("Loading Subject " + str(subject_predict))
        testdata_U = pd.read_csv(file,sep=',',header=None,dtype=np.float32, usecols=[i for i in cols if i not in col_select.astype(int)])
        testfolds_U.append(testdata_U)
        #testdata_U = pd.read_csv(file,sep=',',header=None,dtype=np.float32)
        #testfolds_U.append(testdata_U)
        print('Subject ' + str(subject_predict) + ' loaded: ' + str(len(testdata_U)) + ' windows.')

        nmuscles=int((len(traindata.columns)-1)/spw)
        print('\nEach window is composed by ' + str(len(testdata_U.columns)) + ' samples.')
        print('The number of muscles is ' + str(nmuscles))


# In[16]:

nmuscles=int((len(traindata.columns)-1)/spw) #used for layer dimensions and stride CNNs
#trainfolds = None
#traindata = None
#import gc
#gc.collect()


# In[17]:

import models
from models import *
models._spw = spw
models._nmuscles = nmuscles
models._batch_size = batch_size


# In[18]:

print(models._nmuscles)

#import models
#from models import *
#TEST DIMENSIONS
#models.nmuscles = nmuscles
def testdimensions():
    model = Model23()
    print(model)
    x = torch.randn(32,1,480)
    model.test_dim(x)
 
#testdimensions()


# In[19]:

fieldnames = ['Fold','Acc_L', 'Acc_U',
              'R_0_U','R_1_U',
              'R_0_L','R_1_L',
              'Stop_epoch','Accuracy_dev'] #coloumn names report FOLD CSV
torch.backends.cudnn.benchmark = True

#TRAINING LOOP
def train_test():
    for k in model_select:
        
        table = BeautifulTable()
        avgtable = BeautifulTable()
        fieldnames1 = [model_lst[k],'Avg','Std_dev'] #column names report GLOBAL CSV
        folder = os.path.join(cwd,'Report_'+str(model_lst[k]))
        if not os.path.exists(folder):
            os.mkdir(folder)

        logfilepath = os.path.join(folder,'log.txt')
        logfile = open(logfilepath,"w") 

        with open(os.path.join(folder,'Report_folds.csv'),'w') as f_fold, open(os.path.join(folder,'Report_global.csv'),'w') as f_global:
            writer = csv.DictWriter(f_fold, fieldnames = fieldnames)
            writer1  = csv.DictWriter(f_global, fieldnames = fieldnames1)
            writer.writeheader()
            writer1.writeheader()
            t0 = 0
            t1 = 0
            for i in range(1,nfold+1):
                
                t0 = time.time()
                setSeeds(0)
                
                class Traindataset(Dataset):
                    def __init__(self):
                        self.data=trainfolds[i-1]
                        self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1])) 
                        self.len=self.data.shape[0]
                        self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]]))
                        if (use_cuda):
                            self.x_data = self.x_data.cuda()
                            self.y_data = self.y_data.cuda()
                    def __getitem__(self, index):
                        return self.x_data[index], self.y_data[index]
                    def __len__(self):
                        return self.len
                
                # Classe per il Testset LEARNED
                class Testdataset_L(Dataset):
                    def __init__(self):
                        self.data=testfolds_L[i-1]
                        self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1]))
                        self.len=self.data.shape[0]
                        self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]]))
                        if (use_cuda):
                            self.x_data = self.x_data.cuda()
                            self.y_data = self.y_data.cuda()
                    def __getitem__(self, index):
                        return self.x_data[index], self.y_data[index]
                    def __len__(self):
                        return self.len
                
                # Classe per il Testset UNLEARNED
                class Testdataset_U(Dataset):
                    def __init__(self):
                        self.data=testfolds_U[i-1]
                        self.x_data=torch.from_numpy(np.asarray(self.data.iloc[:, 0:-1]))
                        self.len=self.data.shape[0]
                        self.y_data = torch.from_numpy(np.asarray(self.data.iloc[:, [-1]]))
                        if (use_cuda):
                            self.x_data = self.x_data.cuda()
                            self.y_data = self.y_data.cuda()
                    def __getitem__(self, index):
                        return self.x_data[index], self.y_data[index]
                    def __len__(self):
                        return self.len

                ## DEFINISCO I DATASET DA CARICARE    
                traindataset = Traindataset()    # Train dataset
                testdataset = Testdataset_L()    # Test dataset LEARNED
                testdataset_U = Testdataset_U()    # Test dataset UNLEARNED

                header( model_lst, k, i, traindataset, testdataset, testdataset_U)

                #train_sampler,dev_sampler,test_sampler=dev_shuffle(shuffle_train,shuffle_test,val_split,traindataset,testdataset)
                #train_sampler,dev_sampler,test_val_sampler,test_sampler=data_split(shuffle_train,shuffle_test,val_split,test_val_split,traindataset,testdataset)
                
                ## CREO I SAMPLER PER DIVIDERE IL TRAINDASTASET
                tr_sampler, d_sampler = split_train(val_split, traindataset)
                
                ## LOADERS
                # torch.utils.data.DataLoader deve avere come argomenti:
                    # dataset -> quello che dobbiamo caricare
                    # batch_size -> 'batch_size' definito negli hyperparameter della rete
                    # drop_last = True -> Se la lunghezza del dataset non è divisibile per 'batch_size', 
                    # l'ultimo batch viene eliminato
                # 'dev_loader e 'train_loader' hanno bisogno anche di
                    # sampler = dev_sampler (una certa parte del trainset 
                    # deve essere usata come 'dev_set')
                
                # Training Set
                train_loader = torch.utils.data.DataLoader(traindataset, batch_size = batch_size, 
                                                          sampler = tr_sampler, drop_last = True)
                print('Train Set dimension: ' + str(len(train_loader)))
                # Development Set
                dev_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, 
                                                           sampler = d_sampler, drop_last=True)
                print('Dev Set dimension: ' + str(len(dev_loader)))
                # Testset LEARNED
                test_val_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                                                drop_last=True)
                print('Test Set (L) dimension: ' + str(len(test_val_loader)))
                # Testset UNLEARNED
                test_loader = torch.utils.data.DataLoader(testdataset_U, batch_size = batch_size, 
                                                          drop_last = True)
                print('Test Set (U) dimension: ' + str(len(test_loader)))
                
                modelClass = "Model" + str(k)
                model = eval(modelClass)()
                
                if (use_cuda):
                    model = model.cuda()

                if doTrain:
                    
                    # criterion = nn.BCELoss(size_average=True)
                    criterion = nn.BCELoss(reduction = 'mean')    # Cambiato perché dava un warning (size_average era un vecchio metodo in Pytorch)
                    optimizer = torch.optim.SGD(model.parameters(), lr)    
                    msg = 'Accuracy on test set before training: '+str(accuracy(test_loader, model))+'\n'
                    print(msg)
                    logfile.write(msg + "\n")
                    #EARLY STOP
                    epoch = 0
                    patience = 0
                    best_acc_dev=0
                    while (epoch<maxepoch and patience < maxpatience):
                        running_loss = 0.0
                        for l, data in enumerate(train_loader, 0):
                            inputs, labels = data
                            if use_cuda:
                                inputs, labels = inputs.cuda(), labels.cuda()
                            inputs, labels = Variable(inputs), Variable(labels)
                            y_pred = model(inputs)
                            if use_cuda:
                                y_pred = y_pred.cuda()
                            loss = criterion(y_pred, labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            #print accuracy ever l mini-batches
                            if l % 2000 == 1999:
                                msg = '[%d, %5d] loss: %.3f' %(epoch + 1, l + 1, running_loss / 999)
                                print(msg)
                                logfile.write(msg + "\n")
                                running_loss = 0.0
                                #msg = 'Accuracy on dev set:' + str(accuracy(dev_loader))
                                #print(msg)
                                #logfile.write(msg + "\n")        
                        accdev = (accuracy(dev_loader, model))
                        msg = 'Accuracy on dev set:' + str(accdev)
                        print(msg)
                        logfile.write(msg + "\n")        
                        is_best = bool(accdev > best_acc_dev)
                        best_acc_dev = (max(accdev, best_acc_dev))
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_acc_dev': best_acc_dev
                        }, is_best,os.path.join(folder,'F'+str(i)+'best.pth.tar'), logfile)
                        if is_best:
                            patience=0
                        else:
                            patience = patience+1
                        epoch = epoch+1
                        logfile.flush()
                        
                if doEval:
                    if use_cuda:                        
                        state = torch.load(os.path.join(folder,'F'+str(i)+'best.pth.tar'))
                    else:
                        state = torch.load(os.path.join(folder,'F'+str(i)+'best.pth.tar'), map_location=lambda storage, loc: storage)
                    stop_epoch = state['epoch']
                    model.load_state_dict(state['state_dict'])
                    if not use_cuda:
                        model.cpu()
                    accuracy_dev = state['best_acc_dev']
                    model.eval()
                    
                    acctest = (accuracy(test_loader, model))
                    acctest_val = (accuracy(test_val_loader, model))
                    accs[i-1] = acctest
                    accs_test_val[i-1] = acctest_val
                    
                    precision_0_U,recall_0_U,f1_score_0_U = pre_rec(test_loader, model, 0.0)
                    precisions_0_U[i-1] = precision_0_U
                    recalls_0_U[i-1] = recall_0_U
                    f1_scores_0_U[i-1] = f1_score_0_U
                    
                    precision_1_U,recall_1_U,f1_score_1_U = pre_rec(test_loader, model, 1.0)
                    precisions_1_U[i-1] = precision_1_U
                    recalls_1_U[i-1] = recall_1_U
                    f1_scores_1_U[i-1] = f1_score_1_U
                    
                    precision_0_L,recall_0_L,f1_score_0_L = pre_rec(test_val_loader, model, 0.0)
                    precisions_0_L[i-1] = precision_0_L
                    recalls_0_L[i-1] = recall_0_L
                    f1_scores_0_L[i-1] = f1_score_0_L
                    
                    precision_1_L,recall_1_L,f1_score_1_L = pre_rec(test_val_loader, model, 1.0)
                    precisions_1_L[i-1] = precision_1_L
                    recalls_1_L[i-1] = recall_1_L
                    f1_scores_1_L[i-1] = f1_score_1_L
                    
                    accs_dev[i-1] = accuracy_dev
                    
                    writer.writerow({'Fold': i,'Acc_L': acctest_val, 'Acc_U': acctest,
                                     #'P_0_U': precision_0_U,'R_0_U': recall_0_U,'F1_0_U': f1_score_0_U,
                                     'R_0_U': recall_0_U,
                                     #'P_1_U': precision_1_U,'R_1_U': recall_1_U,'F1_1_U': f1_score_1_U,
                                     'R_1_U': recall_1_U,
                                     #'P_0_L': precision_0_L,'R_0_L': recall_0_L,'F1_0_L': f1_score_0_L,
                                     'R_0_L': recall_0_L,
                                     #'P_1_L': precision_1_L,'R_1_L': recall_1_L,'F1_1_L': f1_score_1_L,
                                     'R_1_L': recall_1_L,
                                     'Stop_epoch': stop_epoch,'Accuracy_dev': accuracy_dev})
                    table.column_headers = fieldnames
                    table.append_row([i,acctest_val,acctest,
                                      #precision_0_U,recall_0_U,f1_score_0_U,
                                      recall_0_U,
                                      #precision_1_U,recall_1_U,f1_score_1_U,
                                      recall_1_U,
                                      #precision_0_L,recall_0_L,f1_score_0_L,
                                      recall_0_L,
                                      #precision_1_L,recall_1_L,f1_score_1_L,
                                      recall_1_L,
                                      stop_epoch,accuracy_dev])
                    print(table)
                    print('----------------------------------------------------------------------')
                    logfile.write(str(table) + "\n----------------------------------------------------------------------\n")
                    t1 = time.time()
                    times[i-1] = int(t1-t0)
                    
                    ## -- Estrai il Basografico
                
                fold_predicted = fold_offset
                if doExtractBaso:
                    if use_cuda:                        
                        state = torch.load(os.path.join(folder,'F'+str(fold_predicted)+'best.pth.tar'))
                    else:
                        state = torch.load(os.path.join(folder,'F'+str(fold_predicted)+'best.pth.tar'), map_location=lambda storage, loc: storage)
                    stop_epoch = state['epoch']
                    model.load_state_dict(state['state_dict'])
                    if not use_cuda:
                        model.cpu()
                    accuracy_dev = state['best_acc_dev']
                    model.eval()
                    
                    ## -- QUESTO DEVI CONTROLLARLO! SE ESCE CHE i È DIVERSO DALLA FOLD DEVI RIFARE TUTTO!
                    print('F'+str(fold_predicted)+'best.pth.tar')
                    predicted_baso = save_predicted_baso(test_loader, model, subject_predict, model_lst[k], fold_predicted)   
            
            duration = str(datetime.timedelta(seconds=np.sum(times)))
            writer.writerow({})
            writer.writerow({'Fold': 'Elapsed time: '+duration})
            avg_acc_test_val = round(np.average(accs_test_val),3)
            std_acc_test_val = round(np.std(accs_test_val),3)
            
            avg_acc_test_val,avg_a,avg_p_0_U,avg_r_0_U,avg_f_0_U,avg_p_1_U,avg_r_1_U,avg_f_1_U,avg_p_0_L,avg_r_0_L,avg_f_0_L,avg_p_1_L,avg_r_1_L,avg_f_1_L,avg_a_d=averages([accs_test_val,accs,precisions_0_U,recalls_0_U,f1_scores_0_U,precisions_1_U,recalls_1_U,f1_scores_1_U,precisions_0_L,recalls_0_L,f1_scores_0_L,precisions_1_L,recalls_1_L,f1_scores_1_L,accs_dev])
            std_acc_test_val,std_a,std_p_0_U,std_r_0_U,std_f_0_U,std_p_1_U,std_r_1_U,std_f_1_U,std_p_0_L,std_r_0_L,std_f_0_L,std_p_1_L,std_r_1_L,std_f_1_L,std_a_d=stds([accs_test_val,accs,precisions_0_U,recalls_0_U,f1_scores_0_U,precisions_1_U,recalls_1_U,f1_scores_1_U,precisions_0_L,recalls_0_L,f1_scores_0_L,precisions_1_L,recalls_1_L,f1_scores_1_L,accs_dev])
            
            writer1.writerow({model_lst[k]: 'Acc_U','Avg': avg_a,'Std_dev': std_acc_test_val})
            writer1.writerow({model_lst[k]: 'Acc_L','Avg': avg_acc_test_val,'Std_dev': std_a})
            writer1.writerow({model_lst[k]: 'P_0_U','Avg': avg_p_0_U ,'Std_dev': std_p_0_U})
            writer1.writerow({model_lst[k]: 'R_0_U','Avg': avg_r_0_U,'Std_dev': std_r_0_U})
            writer1.writerow({model_lst[k]: 'F1_0_U','Avg': avg_f_0_U,'Std_dev': std_f_0_U})
            writer1.writerow({model_lst[k]: 'P_1_U','Avg': avg_p_1_U,'Std_dev': std_p_1_U})
            writer1.writerow({model_lst[k]: 'R_1_U','Avg': avg_r_1_U,'Std_dev': std_r_1_U})
            writer1.writerow({model_lst[k]: 'F1_1_U','Avg': avg_f_1_U,'Std_dev': std_f_1_U})            
            writer1.writerow({model_lst[k]: 'P_0_L','Avg': avg_p_0_L,'Std_dev': std_p_0_L})
            writer1.writerow({model_lst[k]: 'R_0_L','Avg': avg_r_0_L,'Std_dev': std_r_0_L})
            writer1.writerow({model_lst[k]: 'F1_0_L','Avg': avg_f_0_L,'Std_dev': std_f_0_L})
            writer1.writerow({model_lst[k]: 'P_1_L','Avg': avg_p_1_L,'Std_dev': std_p_1_L})
            writer1.writerow({model_lst[k]: 'R_1_L','Avg': avg_r_1_L,'Std_dev': std_r_1_L})
            writer1.writerow({model_lst[k]: 'F1_1_L','Avg': avg_f_1_L,'Std_dev': std_f_1_L})                        
            writer1.writerow({model_lst[k]: 'Acc_dev','Avg': avg_a_d,'Std_dev': std_a_d})
            writer1.writerow({})
            writer1.writerow({model_lst[k]: 'Elapsed time: '+duration})
            avgtable.column_headers = fieldnames1
            avgtable.append_row(['Acc_U',avg_a,std_a])
            avgtable.append_row(['Acc_L',avg_acc_test_val,std_acc_test_val])
            avgtable.append_row(['P_0_U',avg_p_0_U,std_p_0_U])
            avgtable.append_row(['R_0_U',avg_r_0_U,std_r_0_U])
            avgtable.append_row(['F1_0_U',avg_f_0_U,std_f_0_U])
            avgtable.append_row(['P_1_U',avg_p_1_U,std_p_1_U])
            avgtable.append_row(['R_1_U',avg_r_1_U,std_r_1_U])
            avgtable.append_row(['F1_1_U',avg_f_1_U,std_f_1_U])                        
            avgtable.append_row(['P_0_L',avg_p_0_L,std_p_0_L])
            avgtable.append_row(['R_0_L',avg_r_0_L,std_r_0_L])
            avgtable.append_row(['F1_0_L',avg_f_0_L,std_f_0_L])
            avgtable.append_row(['P_1_L',avg_p_1_L,std_p_1_L])
            avgtable.append_row(['R_1_L',avg_r_1_L,std_r_1_L])
            avgtable.append_row(['F1_1_L',avg_f_1_L,std_f_1_L])            
            avgtable.append_row(['Accuracy_dev',avg_a_d,std_a_d])
            print(avgtable)
            logfile.write(str(avgtable) + "\n")
            msg = 'Elapsed time: '+ duration + '\n\n'
            print(msg)
            logfile.write(msg )

        logfile.close()
        


# In[ ]:

nmuscles=int((len(traindata.columns)-1)/spw)
if use_cuda and not use_gputil and cuda_device!=None and torch.cuda.is_available():
    with torch.cuda.device(cuda_device):
        print('I am using CUDA: SUCCESS!')
        train_test()
else:
    print('I am NOT using CUDA: SUCCESS!')
    train_test()


# In[ ]:




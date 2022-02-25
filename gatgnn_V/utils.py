import os, shutil
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import RandomSampler 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import LabelBinarizer
import argparse

from   sklearn.model_selection import train_test_split
from   sklearn.metrics import mean_absolute_error as sk_MAE
from   tabulate import tabulate
import random,time
import torch, seaborn as sns

def k_fold_split(k, idx_array):
    some_arr = np.array_split(idx_array,k)
    return some_arr

def get_dataset(src_folder):
    cifs     = [x for x in os.listdir(src_folder) if x.endswith('.cif')]

    #---- read-in the csv
    electrodes_file = 'DATA/electrodes.csv'
    df              = pd.read_csv(electrodes_file)
    valid_rows      = np.zeros(df.shape[0])

    lowMPID  = (df.low_mpid+'.cif').values.reshape(-1)
    highMPID = (df.high_mpid+'.cif').values.reshape(-1)

    for i  in range(df.shape[0]):
        if lowMPID[i] in cifs and highMPID[i] in cifs:
            valid_rows[i] = 1
    kept     = np.nonzero(valid_rows)[0].reshape(-1)
    df       = df.loc[kept]


    iontype   = df.battery_id.values
    iontype   = [x.split('_')[1] for x in iontype]
    df['ion'] = iontype
    df        = df[['low_mpid','high_mpid','avg_voltage','ion']]

    return df

def torch_MAE(tensor1,tensor2):
    return torch.mean(torch.abs(tensor1-tensor2))

def output_training(metrics_obj,epoch,estop_val,extra='---'):
    header_1, header_2 = 'MSE | e-stop','MAE | TIME'
    if metrics_obj.c_property in ['is_metal','is_not_metal']:
        header_1,header_2     = 'Cross_E | e-stop','Accuracy | TIME'

    train_1,train_2 = metrics_obj.training_loss1[epoch],metrics_obj.training_loss2[epoch]
    valid_1,valid_2 = metrics_obj.valid_loss1[epoch],metrics_obj.valid_loss2[epoch]
    
    tab_val = [['TRAINING',f'{train_1:.4f}',f'{train_2:.4f}'],['VALIDATION',f'{valid_1:.4f}',f'{valid_2:.4f}'],['E-STOPPING',f'{estop_val}',f'{extra}']]
    
    output = tabulate(tab_val,headers= [f'EPOCH # {epoch}',header_1,header_2],tablefmt='fancy_grid')
    print(output)    

def train(net,train_loader1,train_loader2,metrics,optimizer, device):
    # TRAINING-STAGE
    net.train()       
    for i,High_Low_data in enumerate(zip(train_loader1,train_loader2)):
        data0, data1  = High_Low_data
        data0, data1  = data0.to(device), data1.to(device)
        
        predictions  = net(data0, data1)
        train_label  = metrics.set_label('training',data0)
        loss         = metrics('training',predictions,train_label,1)
        _            = metrics('training',predictions,train_label,2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.training_counter+=1 
    return metrics, net 

def evaluate(net,valid_loader1,valid_loader2,metrics, device, with_labels=False):
    # EVALUATION
    net.eval()
    out_labels = [[],[]]
    for i,High_Low_data in enumerate(zip(valid_loader1,valid_loader2)):
        data0, data1  = High_Low_data
        data0, data1  = data0.to(device), data1.to(device)
        with torch.no_grad():
            predictions  = net(data0, data1)
        
        valid_label        = metrics.set_label('validation',data0)

        if with_labels:
            out_labels[0].append(valid_label.detach())
            out_labels[1].append(predictions.detach())
        else:
            _   = metrics('validation',predictions,valid_label,1)
            _   = metrics('validation',predictions, valid_label,2)
            metrics.valid_counter+=1
            
    return metrics, net, out_labels

    # metrics.reset_parameters('validation',epoch)
    # scheduler.step()
    # end_time         = time.time()
    # e_time           = end_time-start_time
    # metrics.save_time(e_time)
    
    # # EARLY-STOPPING
    # early_stopping(metrics.valid_loss2[epoch], net)
    # flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
    # if early_stopping.FLAG == True:    estop_val = flag_value
    # else:
    #     estop_val        = '@best: saving model...'; best_epoch = epoch+1
    # output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

    # if early_stopping.early_stop:
    #     print("> Early stopping")
    #     break




    # # SAVING MODEL
    # print(f"> DONE TRAINING !")
    # shutil.copy2('MODELS/crystal-checkpoint.pt', f'MODELS/{crystal_property}.pt')

        
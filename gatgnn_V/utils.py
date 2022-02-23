import os
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
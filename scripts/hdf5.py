# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:07:20 2014

@author: Yasumasa

def save_hdf5(df, file_name)
This function saves a dataframe in the current working directory 
@param   df          Pandas DataFrame
@param   file_name   User defined file name 
@return  path        Full path to the save HDF5 file


def read_hdf5(path)
This function reads a HDF5 file and return a dataframe
@param   path        Full path to the save HDF5 file
@return  df          Pandas DataFrame

"""

import os
import pandas as pd


def save_hdf5(df, file_name):
    path = os.getcwd() + '\\' + file_name       
    df.to_hdf(path,'df', mode='a', format='table')
    return path

def read_hdf5(path):
    return pd.io.pytables.read_hdf(path,'df')

    

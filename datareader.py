# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:45:12 2020

@author: giamm
"""

from pathlib import Path
import numpy as np
import csv
import math

##############################################################################


# This scripted is used to create methods that properly read files


##############################################################################

# The base path is saved in the variable basepath, it is used to move among
# directories to find the files that need to be read.
basepath = Path(__file__).parent


##############################################################################

def read_param(filename,delimit,dirname):
    
    ''' The function reads from a .csv file in which some parameters are saved. 
    The file has the parameter's name in the first column, its value 
    in the second one and its unit of measure (uom) in the third one.
        
    Inputs:
        filname - string containing the name of the file (extension of the file: .dat)
        delimit - string containing the delimiting element
        dirname - name of the folder where to find the file to be opened and read
        
    Outputs:
        params - dict, containing the parameters (keys) and their values, as entered by 
                 by the user and stored in the .csv file        
    '''
    
    dirname = dirname.strip()

    filename = filename.strip()
    if not filename.endswith('.csv'): filename = filename + '.csv'
    
    fpath = basepath / dirname 
    
    params = {}
    
    try:
        with open(fpath / filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=delimit,quotechar="'")
            
            header_row = 1
            for row in csv_reader:

                if header_row == 1:

                    for ii in range(len(row)): row[ii] = row[ii].lower().strip().replace(' ', '_')
                    header = {
                        'name': row.index('name'),
                        'value': row.index('value'),
                    }

                    header_row = 0
                    continue

                else:

                    param_name = row[header['name']]
                    param_val = row[header['value']]
                    
                    try: 
                        param_val = int(param_val)
                    except: 
                        try: param_val = float(param_val)
                        except: param_val = param_val
                        
                    params[param_name] = param_val
                          
    except:
        print('Unable to open this file')
    
    # print('Im returning params: {}'.format(params))
    return(params)

 
##############################################################################
    
def read_general(filename,delimit,dirname):
    
    ''' The function reads from a .csv file in which the header is a single row
        
    
    Inputs:
        filname - string containing the name of the file (extension of the file: .dat)
        delimit - string containing the delimiting element
        dirname - name of the folder where to find the file to be opened and read
        
    Outputs:
        data - 2d-array containing the values in the file'
    '''
    
    dirname = dirname.strip()

    filename = filename.strip()
    if not filename.endswith('.csv'): filename = filename + '.csv'
    
    fpath = basepath / dirname 
    
    data_list=[]
    
    try:
        with open(fpath / filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=delimit)
            next(csv_reader, None) 
            for row in csv_reader:
                
                data_list.append(row)              
                
    except:
        
        print('')
        # print('Unable to open this file')
    
    # Creating a 2D-array containing the data(time in the first column and power in the second one)
    data = np.array(data_list,dtype='float')
    return(data)



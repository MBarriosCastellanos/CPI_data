# %% =======================================================================
# Import dict
# ==========================================================================
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics
import time                                     # time calculation
from functions import time_elapsed 
# print list, plot sigmoid 
d_s_a = {                               # arguments transfrom in h5 file
  'compression':'gzip', 'shuffle':True, 'fletcher32':True }
print('Basic Libraries imported')
START = time.time()

#%% =======================================================================
# Define keys data
# ==========================================================================
keys = ['AC-01X', 'AC-01Y', 'AC-01Z', 'AC-02',  'AC-03', 'AC-04',
  'AC-05', 'AC-06X', 'AC-06Y', 'AC-06Z']  # keys    
DF = pd.DataFrame(h5todict('data/DF.h5')) # Transform DF to pandas dataframe
t = h5todict('data/t.h5');                t = t['t']          # time vector
curves = list(DF['curve'].unique())       # list curves   
time_elapsed(START)             # Time elapsed in the process
print('Analyses data defined...')

#%% ========================================================================
# Change labeled 
# ==========================================================================
# Changed  AC-02_AC-05, AC-03_AC-04, AC-05x_AC-06x, AC-05y_AC-06y, and z
print('██████████████████ Change label in curves  ███████████████████████')
data = {}
for key in keys:
  path_data = 'data/' + key + '.h5' # path data
  x_dict = h5todict(path_data);     # dict
  key_r =  list(x_dict.keys())[0]   
  print('file = %s.h5, key = %s, shape = %s'%(
    key, key_r, x_dict[key_r].shape))
  if key!=key_r:
    print('file %s.h5, but key %s, key changed to %s'%(key, key_r, key))
    dicttoh5({key:x_dict[key_r]}, path_data, create_dataset_args=d_s_a) 
    time_elapsed(START)             # Time elapsed in the process
  print('constucut vib data dict, key + %s'%key)
  data[key] = x_dict[key_r] # construct big data
del key, x_dict, key_r
time_elapsed(START)                 # Time elapsed in the process

#%% ========================================================================
# create new dict
# ==========================================================================
print('█████████████████████ Create a big data ██████████████████████████')
for i, curve in enumerate(curves):
  print('constucut curves dict, key + %s, %s/%s'%(curve, i+1, len(curves)))
  df = DF[DF['curve']==curve]     # evaluate a data frame
  I = list(df.index)              # get index to split vib
  data_df = {}                    # dict of this experiment
  for key in keys:            # for every vib sensor
    data_df[key] = data[key][I]   # split data for every experiment
  dicttoh5(data_df, 'data/' + curve + '.h5', create_dataset_args=d_s_a)
  time_elapsed(START)             # Time elapsed in the process
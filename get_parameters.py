# %% =======================================================================
# Import dict
# ==========================================================================
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics
import glob                                     # inspect folders
import time                                     # time calculation
from scipy.optimize import minimize, Bounds     # optimization arguments
from functions_signal import add_vib            # add vibration to vector
from functions import sigmoid, dsigmoid, printProgressBar,intt, time_elapsed
#sigmoid function, progress bar, integration, time calculation
from functions import enum_vec#, plot_sigmoid   
# print list, plot sigmoid 
from functions import plot_sigmoid
d_s_a = {                               # arguments transform in h5 file
  'compression':'gzip', 'shuffle':True, 'fletcher32':True }
print('Basic Libraries imported')

#%% ========================================================================
# function to find folders
# ==========================================================================
START = time.time()
from functions_signal import  basic_inf, time_signal, get_var
# determine a series of bound
get_lims = lambda step , maxx, minn: [
  [[i if i!=0 else minn,(i+step)]] for i in np.arange(0,maxx,step)]
# integration function
print('Signal Libraries imported')

# %% =======================================================================
# Define PR data
# ==========================================================================
# possible vibration keys to analyze ---------------------------------------
keys = ['AC-01X', 'AC-01Y', 'AC-01Z', 'AC-02',  'AC-03', 'AC-04',
  'AC-05', 'AC-05X', 'AC-05Y', 'AC-05Z'];  enum_vec(keys) 
i_key = int(input('Please enter a position key:\n'))
key = keys[i_key];  print('the selected key is ███ %s ███'%(key))
# DF_PR signals ------------------------------------------------------------
sig = [ 'time', 'fft', 'welch']  
# statistics to analyze in every signal sig---------------------------------
stat =  [ 'mean', 'var', 'gmean', 'hmean', 'rms', 'crest', 'wmean']
# frequency bound to analyze -----------------------------------------------
var = [''] + get_var(5e3, 10e3) + get_var(2e3, 6e3) + get_var(1e3, 5e3) \
    + get_var(250, 1000)  + get_var(750, 750) + ['1alab', '2alab']                        
lims = [['']] + get_lims(5e3, 10e3, 100) + get_lims(2e3, 6e3, 100) + \
  get_lims(1e3, 5e3, 100) + get_lims(250, 1000, 10) + get_lims(750, 750, 60)
# Construct column names ---------------------------------------------------
columns = ['_'.join([i,j,k]) for k in var  for i in sig  for j in stat] 
# possible signal normalization to analyze ---------------------------------
norms = ['x', 'log']; enum_vec(norms) 
i = int(input('Please enter a normalization kind:\n'))
norm = norms[i]; print('the normalization key is ███ %s ███'%(norm))
# --------------- ----------------------------------------------------------
del norms, i, keys, get_lims, get_var, var
print('Analyses data defined...')
time_elapsed(START)             # Time elapsed in the process

#%% ========================================================================
# Data Frame process variables and Vibration analysis
# ==========================================================================
print('█████████████████ Getting process dataframe  █████████████████████')
DF = pd.DataFrame(h5todict('data/DF.h5'))  # Transform DF to pandas dataframe
X_var = h5todict('data/' + key + '.h5');  X_var = X_var[key]  # vib data
t = h5todict('data/t.h5');                t = t['t']          # time vector
curves = list(DF['curve'].unique())          
time_elapsed(START)             # Time elapsed in the process

#%% ========================================================================
# Sigmoid functions
# ==========================================================================
print('███████████████████ Genering sigmoid function ████████████████████')
# Getting  bound between water-in-oil & transition & oil-in-water ----------
i1 = []; i2 = []            # water-in-oil__i1__transition__i2__oil-in-water
for curve in curves:        # for every curve
  df = DF[DF['curve']==curve] # dataframe for only this curve
  wc = np.array( df['wc']);                   # watercut [%]
  h = np.array(df['dp']*1e5/(df['rho']*9.81)) # pressure head [m]
  # parameter sigmoid function  a0 = h_max, a1 = h_max - h_min, a2 = slope
  a = [h.min(), h[:-1].max() - h.min(), 1, 40]# a3 = water cut transition
  # Getting sigmoid PR through optimization ------------------------
  bounds = Bounds([a[0], a[1]*0.8, 0.05,  0], # optimization bound
    [a[0]*1.2, a[1], 2,  100])                # hmin, hmax - hmin, slope, wc_i
  a = minimize(lambda a: sum((sigmoid(a,wc)-h)**2), a, bounds=bounds,
    method='BFGS').x
  x = np.linspace(0, 100, num=10001)          # water cut possibles
  dy_max = dsigmoid(a, a[3])  # derivate of sigmoid function in wc
  limit = np.where(dsigmoid(a, x)>dy_max*0.2)[0] # transition region
  I = df.index                         # index of this curve
  l1 = np.where(wc<x[ limit[ 0] ])[0][-1];  i1.append(I[l1])  # begin trans
  l2 = np.where(wc>x[ limit[-1] ])[0][ 0];  i2.append(I[l2])  # end   trans
  #plot_sigmoid(x, a, wc, h, l1, l2, a[3])
print('The labeling of the emulsion kind is %s'%(
  all(DF['i1'].unique()==i1) and  all(DF['i2'].unique()==i2)))
#del a, df, wc, h, bounds, Bounds, minimize, dsigmoid, dy_max 
#del x, limit, l1, l2, curve
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# getting dict of signals
# ==========================================================================
print('██████████████████████ Getting signal class ██████████████████████')
w = np.array(DF['w']/60)                            # rotational speed
n_al = 7                                            # blades number
signals = time_signal(X_var, t, w, n_al)            # transf to signal class 
del X_var, t, time_signal
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Creating PR data
# ==========================================================================
print('██████████████████████ Calculating DF_PR PR ██████████████████████')
lim_n1 =  np.c_[w*n_al   - w*2,  w*n_al   + w*2]  # rotational harmonic sid
lim_n2 =  np.c_[w*n_al*2 - w*4,  w*n_al*2 + w*4]  # blade harmonic sidebands
path_PR =  'results/%s_%s.h5'%(key, norm)  # path to save data
if len(glob.glob(path_PR))== 0:                   # if results exist
  bounds = lims if i_key>11 else  lims + [lim_n1] + [lim_n2] # list lims
  len_b = len(bounds)                             # boundary len
  prefix = '         %s/%s '%(0,len_b)            # prefix to progress bar
  printProgressBar(0, len_b, prefix = prefix)
  for b, bound in enumerate(bounds):        # in every boundary
    prefix = '         %s/%s '%(b+1,len_b)
    if len(bound)==1:                       # for the first boundary
      col = signals.sts(sig,  bound[0], norm, stat) # column
      printProgressBar(b + 1, len_b, prefix = prefix)
    else:                     # others bound
      for i, bdy in enumerate(bound):
        printProgressBar((i +1)/len(bound) + b, len_b, prefix = prefix)
        row1 = signals.sts(sig, bdy, norm, stat, 
          split=[i, i+1])     # every column
        col = row1 if all(bdy==bound[0]) else np.r_[col, row1]
    PR = col if bound==bounds[0] else np.c_[PR, col] # save parameters
  PR = pd.DataFrame(PR, columns=columns)

  dicttoh5(PR.to_dict('list'), path_PR, create_dataset_args=d_s_a)
  del bounds
else:
  PR = pd.DataFrame(h5todict(path_PR))
  print('not Calculating measurement metrics, data path')
DF_PR = pd.concat([DF, PR], axis=1)
time_elapsed(START)             # Time elapsed in the process

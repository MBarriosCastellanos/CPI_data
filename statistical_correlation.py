# %% =======================================================================
# Import dict
# ==========================================================================
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics
import glob                                     #  inspect folders
import matplotlib.pyplot as plt                 # Plot several images
from functions import printProgressBar, create_folder
from functions import enum_vec, time_elapsed
import time                                     # time calculation
plt.rcParams['figure.figsize'] = (36,20)
print('Basic Libraries imported █████████████████████████████████████████')

#%% ========================================================================
# function 
# ==========================================================================
START = time.time()
from functions_signal import get_columns, get_var, get_z 
from functions_signal import Colormesh, plot_columns#, plot_curves
print('Function created  ████████████████████████████████████████████████')

# %% =======================================================================
# Define parameter data
# ==========================================================================
fl = ''      # vib_ace, vib_vel, vib_dez, process
plot_kind = 'lin'   # log or lin
# possible vibration keys to analyze ---------------------------------------
keys = ['AC-01X', 'AC-01Y', 'AC-01Z', 'AC-02',  'AC-03', 'AC-04',
  'AC-05', 'AC-05X', 'AC-05Y', 'AC-05Z'];  enum_vec(keys) 
i_key = int(input('Please enter a position key:\n'))
key = keys[i_key];  print('the selected key is ███ %s ███'%(key))
# DF_PR signals ------------------------------------------------------------
sig = [ 'time', 'fft', 'welch']  
# statistics to analyse in every signal sig---------------------------------
stat =  [ 'mean', 'var', 'gmean', 'hmean', 'rms', 'crest', 'wmean']
# frequency boundaries to analyze ------------------------------------------
var = [''] + get_var(5e3, 10e3) + get_var(2e3, 6e3) + get_var(1e3, 5e3) \
    + get_var(250, 1000)  + get_var(750, 750) + ['1alab', '2alab']         
# possible signal normalization to analyze ---------------------------------
norms = ['x', 'log']; enum_vec(norms) 
i = int(input('Please enter a normalization kind:\n'))
norm = norms[i]; print('the normalization key is ███ %s ███'%(norm))
# --------------- ----------------------------------------------------------
del norms, i, keys, get_var
print('Analyses data defined...')
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# folder creations and get DF
# ==========================================================================
print('███████████████████ Creation folder for images ███████████████████')
folder_r = 'results'                       # folder results
folder_i = 'images/' +  key                       # folder images
folder_s = folder_i + '/' + fl + '_' + plot_kind  # folder plotkind
folder_is = folder_s + '/' + norm                 # folder to save images
path_pr = '/'.join([                              # PR data
  folder_r, fl, key + '_' + norm + '.h5'])   
# --------------------------------------------------------------------------
create_folder(folder_i)                          # images/AC-02   example
create_folder(folder_s)                          # images/AC-02/vib_ace_lin
create_folder(folder_is)                         # images/AC-02/.../x  
create_folder(folder_is + '/spec' )              # images/AC-02/.../x/spec
for sg in sig: create_folder(folder_is + '/' + sg) # images/AC-02/.../x/fft
#for v in var[1:]: create_folder(folder_is + '/' + v# images/AC-02/x/0-250hHz
# --------------------------------------------------------------------------
DF = pd.DataFrame(h5todict('data/DF.h5'))  # Basic exp inf
PR = pd.DataFrame(h5todict(path_pr))              # PR results
DF_PR = pd.concat([DF, PR], axis=1)               # DF concatenate Parameters
wc = (DF_PR.sort_values(by=['wc'])['wc']).unique() # x
curves =  DF_PR['curve'].unique()                  # y
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# compare different experiments and variations
# ==========================================================================
print('████████████████████████████ Save fig ████████████████████████████')
styles = ['o', 'x', '+', 'd', 'v', '*', '|']
li = len(var[1:]); lj = len(sig); lk = len(stat); pre = ' compare curves:'
printProgressBar(0,li, prefix = pre)
for i, vr in enumerate(var[1:]):  # in every variation 0-250hHz...
  for j, sg in enumerate(sig):    # in every signal time, fft, welch
    for k, st in enumerate(stat): # in evert stats gmean, rms, etc...
      printProgressBar(i + (j + (k+1)/lk)/lj, li, prefix = pre)
      columns = get_columns(sg, st, ['', vr])
      plot_columns(DF_PR, curves, columns, path=folder_is + '/' + sg, 
      plot_kind=plot_kind)
      #plot_curves(DF_PR, curves, columns, vr, key, norm)
      plot_columns(DF_PR, curves, columns, path=folder_is + '/' + sg, 
        plot_kind=plot_kind, xi='Ta', styles=styles)
      #plot_curves(PR, curves, columns, vr, key, norm, 
      # xi='Ta', styles=styles)
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Plot spectrums
# ==========================================================================
print('███████████████████████████ Save spec ████████████████████████████')
plt.rcParams['figure.figsize'] = (36,10)
zs = list(DF_PR.columns)                        # variables to plot
for i in ['curve', 'path', 'exp']: zs.remove(i) # Remove variables not plot
y_ticks = [curve[6:] for curve in curves]       # names of curves
y = np.arange(len(curves))                      # curves
printProgressBar(0, len(zs), prefix = ' Plotting:')
for i, z in enumerate(zs):
  j = 'spec' if i<27 else z.split('_')[0]       # folder to save data
  path = '/'.join([folder_is, j, z + '.png'])   # path folder
  printProgressBar(i+1, len(zs), prefix = ' Plotting:')
  if len(glob.glob(path))<1:                    # If figure not exist
    plt.clf();  plt.close();  figure = plt.figure()
    figure.add_subplot(121) #------------------ Figure 1 -------------------
    Ta = get_z(DF_PR, wc, curves)               # Dimensionaless torque
    Colormesh(wc, y, Ta, y_ticks)         # Colormesh x=wc, y=curves, Z=Ta
    figure.add_subplot(122) #------------------ Figure 2 -------------------
    Z = get_z(DF_PR, wc, curves, z_i=z)   # Dimensionaless torque
    Z = np.log10(np.abs(Z)) if plot_kind=='log' else Z  # change log scale
    Colormesh(wc, y, Z, y_ticks, zlabel=z)# Colormesh Z=analysis variable
    figure.tight_layout();    plt.savefig(path)
time_elapsed(START)             # Time elapsed in the process

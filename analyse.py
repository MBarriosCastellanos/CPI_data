# %% =======================================================================
# Import dict
# ==========================================================================
from click import style
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics
import glob                                     #  inspect folders
import matplotlib.pyplot as plt                 # Plot several images
from functions import printProgressBar, create_folder, dft
from functions import sigmoid, dsigmoid, plot_sigmoid
from functions import enum_vec, time_elapsed, enum_vec, filter_nan
from functions_signal import get_z, Colormesh, Norm2, plot_lims
from sklearn import metrics, feature_selection  # metrics decision tree
from tikzplotlib import save as tikz_save
import time                                     # time calculation
from matplotlib import cm                       # cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds     # optimization arguments
from scipy.stats import linregress
plt.rcParams['figure.figsize'] = (12,8)
print('Basic Libraries imported █████████████████████████████████████████')

#%% ========================================================================
# function 
# ==========================================================================
START = time.time()
from functions_signal import get_columns, get_var, get_z, time_signal
from functions_signal import Colormesh, plot_columns#, plot_curves
print('Function created  ████████████████████████████████████████████████')

# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
curves = [i[5:-3] for i in glob.glob('data/s*')]; enum_vec(curves) 
keys = [i[5:-3] for i in glob.glob('data/AC*')];  enum_vec(keys) 
# DF_PR signals ------------------------------------------------------------
sig = [ 'time', 'fft', 'welch']  
# statistics to analyse in every signal sig---------------------------------
stat =  [ 'mean', 'var', 'gmean', 'hmean', 'rms', 'crest', 'wmean']
# frequency bound to analyze -----------------------------------------------
var = [''] + get_var(5e3, 10e3) + get_var(2e3, 6e3) + get_var(1e3, 5e3) \
    + get_var(250, 1000)  + get_var(750, 750) + ['1alab', '2alab']                        
# Construct column names ---------------------------------------------------
columns = ['_'.join([i,j,k]) for k in var  for i in sig  for j in stat] 
# possible signal normalization to analyze ---------------------------------
norms = ['x', 'log'];   enum_vec(norms) 
# --------------- ----------------------------------------------------------

norm = norms[0]; print('the normalization key is ███ %s ███'%(norm))
#del norms, i, keys, get_var, var
DF = pd.DataFrame(h5todict('data/DF.h5'))
print('Analyses data defined...')
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Curve  of h vs wc
# ==========================================================================
label1 = lambda w, mu, q : 'rpm = ' + str(round(w)) + ', ' + \
  '$\mu$ = ' + str(round(mu,1)) + ' [mPa-s], ' + '$Q$ = ' + \
  str(round(q,1)) +' [$m^3/h$]'
curves_sel = ['s_2ph_31c_3000rpm_433m2h', 's_2ph_28c_2400rpm_261m3h',
  's_2ph_30c_2400rpm_276m3h', 's_2ph_30c_1800rpm_204m3h']
styles = ["o-.", "s--", "d--", ".:"]
for i, curve in enumerate(curves_sel):
  df = DF[DF["curve"]==curve]
  label = label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean())
  plt.plot(df["wc"], df["dp"]*1e5/df["rho"]/9.81/7, styles[i], 
    label=label, color="k")
plt.xlabel('water cut %'); plt.xlim([0,100])
plt.ylabel('h [m]'); plt.ylim([2,12])
plt.legend(loc='upper left')

# %% =======================================================================
# Curve  of Ta vs wc
# ==========================================================================
for i, curve in enumerate(curves_sel):
  df = DF[DF["curve"]==curve]
  label = label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean())
  plt.plot(df["wc"], df["Ta"]/7, styles[i], 
    label=label, color="k")
plt.xlabel('water cut %'); plt.xlim([0,100]); plt.yscale('log')
plt.ylabel('dimesionaless torque')#; plt.ylim([2,12])
plt.legend(loc='upper right')

#%% ========================================================================
# Sigmoid functions
# ==========================================================================

# Getting  bound between water-in-oil & transition & oil-in-water ----------
print('███████████████████ Genering sigmoid function ████████████████████')
curve = curves[6]        # for every curve
df = DF[DF['curve']==curve] # dataframe for only this curve
wc = np.array( df['wc']);                   # watercut [%]
h = np.array(df['dp']*1e5/(df['rho']*9.81)) # pressure head [m]
# parameter sigmoid function  a0 = h_max, a1 = h_max - h_min, a2 = slope
a = [h.min(), h[:-1].max() - h.min(), 1, 40]# a3 = water cut transition
# Getting sigmoid PR through optimization ------------------------
bounds = Bounds([a[0], a[1]*0.8, 0.05,  0], # optimization bound
  [a[0]*1.2, a[1], 2,  100])        # hmin, hmax - hmin, slope, wc_i
a = minimize(lambda a: sum((sigmoid(a,wc)-h)**2), a, bounds=bounds).x
x = np.linspace(0, 100, num=10001)          # water cut possibles
dy_max = dsigmoid(a, a[3])  # derivate of sigmid function in wc
limit = np.where(dsigmoid(a, x)>dy_max*0.2)[0] # transition region
l1 = np.where(wc<x[ limit[ 0] ])[0][-1]# begin trans
l2 = np.where(wc>x[ limit[-1] ])[0][ 0]# end   trans
plot_sigmoid(x, a, wc, h, l1, l2, a[3])
plt.title(label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean()))
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
curve = curves[13];  print('the selected curve is ███ %s ███'%(curve))
df = DF[DF['curve']==curve]
i1 = np.where(df.index==int(df['i1'].min()))[0][0]
i2 = np.where(df.index==int(df['i2'].min()))[0][0]
t = h5todict('data/t.h5')['t']
vib = h5todict('data/' + curve + '.h5') 
for key in keys:
  signal = time_signal(vib[key], t, df['w'], 7)
  vib_arr = np.dstack((vib_arr, signal.Y)) if key!=keys[0] else signal.Y
wc1 = df['wc'].loc[int(df['i1'].min())];  wct1 = wc1/3 
wc2 = df['wc'].loc[int(df['i2'].min())];  wct2 = (100-wc2)/3
I = []
for wc in [wct1, wct1*2, 0.5*(wc1+wc2), wc2 + wct2, wc2 + wct2*2]:
  I.append(np.where(df['wc']>=wc)[0][0])


# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
n = np.where(signal.f>1000)[0][0]
x = np.array(df['wc'])
y = signal.f[:n:512]
z = np.log10(vib_arr[:, :n:512, :])
X, Y = np.meshgrid(x,y)
lims = [0, i1 + 1, i2, len(x) + 1]
labels = ["water-in-oil", "transition", "oil-in-water"]
markers = ["o", "|", "s"]
fig1, ax1 = plt.subplots(2, 2, constrained_layout=True)
fig2, ax2 = plt.subplots(2, 3, constrained_layout=True)
axes = [i for i in ax2.flat][:3] + [
  i for i in ax1.flat] + [i for i in ax2.flat][3:]
for j, ax in enumerate(axes):
  for i, m in enumerate(markers):
    cs = ax.scatter(X[:, lims[i]:lims[i+1]], Y[:, lims[i]:lims[i+1]], 
      c=z[lims[i]:lims[i+1],:,j].T,  marker=m,
      cmap=cm.copper, vmin=-5.5, vmax=-3.5, alpha=0.8
    );                          ax.set_title(keys[j])
  ax.set_xlim([0, 100]);        ax.set_ylim([0, 1000])
  ax.set_xlabel('water cut %'); ax.set_ylabel('frequency [Hz]')
columns = [plt.plot([], [], markers[i], markerfacecolor='w',
                    markeredgecolor='k')[0] for i in range(3)]
fig1.colorbar(cs, ax=ax1.flat[3], location="right")
fig1.legend(columns, labels, loc='upper left', bbox_to_anchor=(0.92, 0.8))
fig2.colorbar(cs, ax=ax2.flat[5], location="right")
fig2.legend(columns, labels, loc='upper left', bbox_to_anchor=(0.92, 0.8))
#tikz_save("images/" + key + '.tex')

# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
plt.rcParams['figure.figsize'] = [7,7]
plt.rcParams["figure.autolayout"] = True
x = np.array(df['wc'])
y = signal.f[:n]
for i,j in [(0,3), (3,7), (7,10)]:
  z = np.arange(len(keys[i:j]))
  X, Y, Z = np.meshgrid(x, y, z)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_box_aspect(aspect = (1,1,2))
  ax.view_init(25, 45)
  cs = ax.scatter3D(X, Y, Z, c=np.log10(vib_arr[:,:n,i:j].T), 
    alpha=0.8, marker='.', cmap=cm.copper, vmin=-5.5, vmax=-3.5
    )
  ax.set_xlabel('Water cut %')
  ax.set_ylabel('Frequency [Hz]')
  ax.set_zticks(z)
  ax.set_zticklabels(keys[i:j])
  plt.colorbar(cs)
  plt.tight_layout()
  plt.show()

  
# %% =======================================================================
# Pearson Correlation
# ==========================================================================
key = keys[3];  
for key in keys:
  print('the selected key is ███ %s ███'%(key))
  norm = 'log'
  path_pr = '/'.join([                          # PR data
    'results', key + '_' + norm + '.h5'])   
  PR = pd.DataFrame(h5todict(path_pr))          # PR results
  DF_PR = pd.concat([DF, PR], axis=1)           # DF concatenate Parameters
  pr_i = 'fft_'       # different parameters to analyse in pearson cor
  j_pr = [i for i in PR.columns if i[:len(pr_i)]==pr_i] #columns wit fft_
  y = DF['Ta']        # vertical axis value
  X = filter_nan(PR[j_pr])      # filter nan and inf values to pearson
  #X = mul(poly(DF['w'].to_numpy()/60, A)**-1, X*A[0])
  R = feature_selection.r_regression(X.to_numpy(), 
    y.to_numpy())     # compute the pearson R for every column of df
  i = abs(R).argsort()[-6:]     # get the six more important R values
  values = abs(R)[i]*100        # get  values to plot
  features = X.columns;   features = features[i]  # get features no 
  # Compare features with R value ---------------------------------------------
  fig, ax = plt.subplots()      # figure with pearson correlation bar
  y1 = ax.barh(features, values, color='dimgray') # figure with barh
  ax.set_xlim([ values.min(), values.max() ])     # x lims
  ax.bar_label(y1, color='black', fmt='%.4f');    plt.show()      # format
  # get Data Frame -----------------------------------------------------------
  plt.plot(y, X[features[-1]], '.', color='dimgray')  # plot best feature
  plt.xlabel('$T_{a}$'); plt.ylabel(features[-1]);    plt.show()  # format

# %% =======================================================================
# Pearson Correlation
# ==========================================================================
for key in keys:
  print('the selected key is ███ %s ███'%(key))
  norm = 'log'
  path_pr = '/'.join([                          # PR data
    'results', key + '_' + norm + '.h5'])   
  PR = pd.DataFrame(h5todict(path_pr))          # PR results
  z = 'fft_rms_0_2hHz'                          # easy limits plot for analysis
  #A = plot_lims(DF, PR, 'lin', z, deg=1, title=z)   # 
  #A = plot_lims(DF, PR, 'lin', z, deg=1, norm=1, title=z)
  fig, axs = plt.subplots(3, 1)
  for ax, j  in zip(axs, [30, 40, 50]):
    i = DF[DF['class']!=0][np.round(DF['w']/60)==j].index
    x = np.array(PR['fft_rms_0_2hHz' ].loc[i])
    y = np.array(DF['class'].loc[i])
    y_imp = np.empty((0))
    x_imp = np.linspace(x.min(), x.max(), num=10000)
    for i in x_imp:
      y_imp = np.r_[y_imp, impurity(y, i)]
    ax.plot(x, y, 'o')
    i_imp = np.where(y_imp==np.min(y_imp))[0]
    ax2 = ax.twinx()
    ax2.plot(x_imp, y_imp, '--', color='orange')
    ax2.plot(x_imp[i_imp], y_imp[i_imp], '.', color='green')
    bounds = Bounds([x.min()], [x.max()])
    fun = lambda t: impurity(y,t[0])
    opt = minimize(fun, [x.mean()], bounds=bounds, method='powell')
    threshold = opt.x[0]
    lb, ub = (np.nanmin(y_imp)*0.98, 2.1)
    ax.vlines(x_imp[i_imp[ [0, int(len(i_imp)/2), len(i_imp)-1] ]],
      lb, ub, color='black', linestyles=':')
    ax.vlines([threshold], lb, ub, color='black')
    ax.set_ylim([lb, ub])
  plt.show()


# %%
# ==========================================================================
def likelihood (y, values):
  ''' likelihood of every value of (values) in y numpy array'''
  s = len(y) if len(y)!=0 else 1  # len of y corrected by zero division
  return np.array([len(np.where(y==v)[0])/s for v in values])
def gini (y, classes, weights=np.empty((0))):
  '''considering the (y) np array, encouter the gini coefficient for 
  classes and the weights '''
  weights = np.ones(len(classes)) if len(weights)==0 else weights
  return 1 - np.sum((likelihood(y, classes)*weights)**2)
def entropy (y, classes, weights=np.empty((0))):
  '''considering the (y) np array, encouter the gini coefficient for 
  classes and the weights '''
  weights = np.ones(len(classes)) if len(weights)==0 else weights
  p = np.abs(likelihood(y, classes)*weights)
  return np.sum(p/(1-p))
def impurity(y, threshold, weighted=False, kind='gini'):
  '''y impurity divided in threshold based on gini criteria'''
  c, mj = np.unique(y, return_counts=True)    # classes
  w = np.sum(mj)/(len(mj)*mj) if weighted==True else np.ones(len(c)) #weights
  left = np.where(x<=threshold)[0];   right = np.where(x>threshold)[0]
  fun = gini if kind=='gini' else entropy
  return np.sum([len(i) /len(y)*fun(y[i], c, w) for i in [left, right]])


# %%


# %%

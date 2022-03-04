# %% =======================================================================
# Import dict
# ==========================================================================
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics\
import glob                                     #  inspect folders
import matplotlib.pyplot as plt
from sympy import rotations                 # Plot several images
from functions import sigmoid, dsigmoid, plot_sigmoid
from functions import enum_vec, time_elapsed, enum_vec, filter_nan
from sklearn import metrics, feature_selection  # metrics decision tree
from tikzplotlib import save as tikz_save       # save latex
import time                                     # time calculation
from matplotlib import cm                       # cm
from scipy.optimize import minimize, Bounds     # optimization arguments
from scipy.stats import linregress              # linear regression
from scipy.signal import find_peaks             # peaks in spectrum
from numpy.random import default_rng            # random number for train
plt.rcParams['figure.figsize'] = (12,8)
print('Basic Libraries imported █████████████████████████████████████████')

#%% ========================================================================
# function 
# ==========================================================================
START = time.time()
from functions_signal import get_var, time_signal, plot_lims
label1 = lambda w, mu, q : '%.0f [Hz], %.0f [mPa-s], %.0f \
  [$\mathrm{m^3/h}$]'%(w/60, mu, q)
def plot1(curves, DF,  x='wc', y='h', y_c=1e5/(9.81*7) ,
  xlabel='water cut %', ylabel='h [m]', loc='best', xlim=[0, 100]):
  '''simples dataframe plot by curves'''
  styles = ['o-.', 's--', 'd--', '.:']*10
  for i, curve in enumerate(curves):
    df = DF[DF['curve']==curve] # used dataframe
    label = label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean())
    df_y = df['dp']/df['rho'] if y=='h' else df[y]
    plt.plot(df[x], df_y*y_c, styles[i], label=label, color='k')
  plt.xlabel(xlabel); plt.ylabel(ylabel); plt.xlim(xlim); plt.legend(loc=loc)
def scatter3d (x, y, z, keys, lims, rows=2, cols=2, 
  xlim=[0, 100], ylim=[0, 1000]):
  '''Simples plots for  fft vibration '''
  X, Y = np.meshgrid(x,y)                 # create a matrix array
  fig, axs = plt.subplots(rows, cols, constrained_layout=True)
  mks = ['o', '|', 's']                                 # markers
  lbs = ['water-in-oil', 'transition', 'oil-in-water']  # labels
  for j, ax in enumerate(axs.flat):       # for in sensors (plots)
    for i, m in enumerate(mks):           # for in emulsion kinds (lbs)
      [xp, yp, zp] = [k[:, lims[i]:lims[i+1]] for k in [X, Y, z[:,:,j].T]]
      cs = ax.scatter(xp, yp, c=zp,       #plot evert emulsion
        marker=m, cmap=cm.copper, vmin=-5.5, vmax=-3.5, alpha=0.8)
    ax.set_title(keys[j]); ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('water cut %'); ax.set_ylabel('frequency [Hz]')
  fig.colorbar(cs, ax=axs.flat[-1], location='right') # colorbar
  fig.legend(lbs, loc='upper left', bbox_to_anchor=(0.92, 0.8)) #legend
  return fig  #return figure
def spectrum(f, fft, df, I, j, n2, keys, n1=0):
  fig, axs = plt.subplots(len(I), 1, sharex=True, sharey=True)
  m = fft[I, n1:n2, j].max();  key = keys[j]      # max vibration
  gc = (.8,.8,.8,); pc = (.3,.3,.3,)  # Grid Color, spectrum color
  for ax, i in zip(axs, I):
    wc = df['wc'].iloc[i]; w = df['w'].iloc[i]/60
    # harmonics ------------------------------------------------------------
    ax.vlines([w*i for i in range(1, 7)] , 0,m*9/8, color=gc, linestyle=':')
    ax.vlines([w*7, w*14] , 0, m*9/8, color=gc, linestyle='--')
    ax.set_yticks([np.round(m*k/4,5) for k in range(1,5)])  # ticks
    # peaks ----------------------------------------------------------------
    peaks, _ = find_peaks(fft[i, n1:n2, j], height=0.001)
    ax.plot(f[peaks], fft[i, peaks, j], 'x', color='k')
    # log plot -------------------------------------------------------------
    ax2 = ax.twinx();           ax2.set_yscale('log')
    ax2.plot(f[n1:n2], fft[i, n1:n2, 1], alpha=0.6, color=pc)
    ax2.set_ylim([6e-5, 4e-1]); ax2.set_yticks([1e-4, 1e-3, 1e-2, 1e-1 ])
    # log plot -------------------------------------------------------------
    ax.plot(f[n1:n2], fft[i, n1:n2, j],  label=('wc = %.2f%%'%wc), color=pc)
    ax.set_ylim([0, m*9/8]);  ax.set_xlim([f[n1], f[n2]]);  
    ax.legend(loc='upper left')
    ax.set_ylabel('a [$\mathrm{m/s^2}$]')
    ax2.set_ylabel('a [$\mathrm{m/s^2}$]', alpha=0.5)

  ax.set_xlabel('frequency [Hz]'); fig.subplots_adjust(hspace=0)
  fig.suptitle(key)
  return fig
def pearson_bar(DF, PR, y_i, ylabel, ini='', end='', n=15):
  '''function to plot pearson correlation'''
  pr = PR.columns
  # filter by init = fft_ and end = AC-05 ----------------------------------
  pr = [i for i in pr if i[:len(ini) ]==ini] if ini !='' else pr
  pr = [i for i in pr if i[-len(end):]==end] if end !='' else pr
  # ------------------------------------------------------------------------
  y = DF[y_i]                 # vertical axis value
  X = filter_nan(PR[pr])      # filter nan and inf values to pearson
  R = feature_selection.r_regression(X.to_numpy(), 
    y.to_numpy())     # compute the pearson R for every column of df
  i = abs(R).argsort()[-n:]     # get the fifteen more important R values
  values = abs(R)[i]*100        # get  values to plot
  features = X.columns;   features = features[i]  # get features no 
  # Compare features with R value ------------------------------------------
  fig, ax = plt.subplots(1, 2)      # figure with pearson correlation bar
  y1 = ax[0].barh(features, values, color='k') # figure with barh
  ax[0].set_xlim([ values.min()*0.99, values.max()*1.01 ])     # x lims
  ax[0].bar_label(y1, color='k', fmt='%.2f')
  ax[0].set_xlabel('Pearson R Coefficient')
  # get Data Frame ---------------------------------------------------------
  ax[1].plot(y, X[features[-1]], '.', color='k')  # plot best feature
  ax[1].set_xlabel(ylabel); ax[1].set_ylabel(features[-1]);   # format
  return fig, features
logit = lambda b, x : 1/ (1 + np.exp( -(b[0] + b[1]*x) ) )
def cost(b, x, y):
  '''cost function to logit regression'''
  p = logit(b, x)
  c = - (1/len(x)) * (y*np.log(p) + (1 - y)*np.log(1 - p) )
  i = np.where(np.isnan(c))[0]; c[i] = 1.0
  return np.sum(c)
def scatter(DF, PR, curves, pr_i, key):
  '''scatter of curves and class'''
  x = y = z =np.empty((0,));      labels = []                 # empty
  for j, curve in enumerate(curves):      # for every curve 
    df = DF[DF['curve']==curve]           # dataframe particular
    labels.append(label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean()))
    x = np.r_[x, PR[pr_i].iloc[df.index]] # parameter
    y = np.r_[y, j*np.ones(len(df))   ]   # curves
    z = np.r_[z, df['class']]             # classes
  fig, ax = plt.subplots(1, 1) # plot best features ------------------------
  colors = [(.6, .6, .6), (.5, .5, .5), (.0, .0, .0)] # colors for 
  for i, m in enumerate(['_', 'o', 's']):   # for every class  
    I = np.where(z==i)[0]                   # index
    ax.plot(x[I], y[I], m, color=colors[i]) # plot 
  ax.set_xlabel(pr_i);                ax.set_title(key)
  ax.set_yticks(range(len(curves)));  ax.set_yticklabels(labels)
  ax.legend(['transition', 'water-in-oil', 'oil-in-water'])
  return fig
def plot_logit(LG, bs, key, feat):
  fig, axs = plt.subplots(3, 1, sharex=True);
  fig.suptitle(key,x=0.2, y=0.93, fontsize=14)
  for ax, w, b, in zip(axs, [30, 40, 50], bs):
    # plot training and test data ------------------------------------------
    lg = LG[LG['w']==w];          t = lg['threshold'].mean()
    train =lg[lg['train'] == 1];  test = lg[lg['train'] == 0]
    ax.plot(train['x'], train['y'], 's', color='k', alpha=0.5)  # train data
    ax.plot(test['x'], test['y'], '.', color='k')                 # test data
    ax.set_xlim([LG['x'].min(), LG['x'].max()]);  ax.set_ylim([-0.05, 1.05])
    # plot logit function --------------------------------------------------
    x_l = np.linspace(LG['x'].min(), LG['x'].max(), num=1000) # parameter
    p_l = logit(b, x_l)                 # likelihood through logit function
    ax.plot(x_l, p_l, '--', color='k')        # plot logit
    ax.vlines(t,-0.1, 1.1, color='gray', linestyles=':')
    ax.set_ylabel('class at %s [Hz]'%w);  
    ax.set_yticks([0, 0.5, 1]); ax.grid('on'); 
  train_c = LG[LG['train'] == 1]; test_c =  LG[LG['train'] == 0]
  y_train_p = pred(train_c['x'], train_c['threshold'])
  y_test_p = pred(test_c['x'], test_c['threshold'])
  fig.text(0.25, 0.9, 'train score =  %.4f\ntest score = %.4f '%( 
    score(train_c['y'], y_train_p), 
    score(test_c['y'], y_test_p)), fontsize=12)
  ax.set_xlabel(feat, fontsize=12); fig.subplots_adjust(hspace=0)
  fig.legend(['train', 'test', 'logit function'], 
    fontsize=12, bbox_to_anchor=(0.9, 0.95), ncol=3)
  return fig
print('Function created  ████████████████████████████████████████████████')

# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
print('██████████████████████████████ Curves █████████████████████████████')
curves = [i[5:-3] for i in glob.glob('data/s*')]; enum_vec(curves) 
print('███████████████████████████████ Keys ██████████████████████████████')
keys = [i[5:-3] for i in glob.glob('data/AC*')];  enum_vec(keys) 
# DF_PR signals ------------------------------------------------------------
print('██████████████████████████████ Signals ████████████████████████████')
sig = [ 'time', 'fft'] ;                 enum_vec(sig) 
# statistics to analyse in every signal sig---------------------------------
print('██████████████ Statistical analysis for every signal ██████████████')
stat =  [ 'mean', 'var', 'gmean', 'hmean', 'rms']
enum_vec(stat) 
# frequency bound to analyze -----------------------------------------------
print('████████████████ Frequency splits for every signal ███████████████')
var = get_var(5e3, 5e3) + get_var(2e3, 6e3) + get_var(1e3, 5e3) \
    + get_var(250, 1000)  + get_var(750, 750) + ['1alab']   
enum_vec(var)                      
# Construct column names ---------------------------------------------------
columns = ['_'.join([i,j,k]) for k in var  for i in sig  for j in stat] 
# possible signal normalization to analyze ---------------------------------
print('███████ Every signal could be normalized by linear or log ████████')
norms = ['x', 'log'];   enum_vec(norms) 
norm = norms[1]; print('the normalization key is ███ %s ███'%(norm))
# --------------------------------------------------------------------------
DF = pd.DataFrame(h5todict('data/DF.h5'))
print('Analyses data defined...')
time_elapsed(START)                           # Time elapsed in the process

# %% =======================================================================
# Initial Curves
# ==========================================================================
print('█████████████████████ wc vs h, and wc vs Ta ██████████████████████')

curves_sel = [curves[i] for i in [9, 2, 6, 5]]
# Curve  of h vs wc --------------------------------------------------------
plot1(curves_sel, DF, loc='upper left');  plt.ylim([2,12]); plt.show()   
# Curve  of Ta vs wc -------------------------------------------------------
plot1(curves_sel, DF, y='Ta', y_c=1/7, ylabel='dimesionaless torque', 
  loc='upper right');   plt.show()
time_elapsed(START)             # Time elapsed in the process

#%% ========================================================================
# Sigmoid functions
# ==========================================================================
# Getting  bound between water-in-oil & transition & oil-in-water ----------
print('███████████████████ Genering sigmoid function ████████████████████')
df = DF[DF['curve']==curves[6]] # dataframe for only this curve
wc = np.array( df['wc']);                   # watercut [%]
h = np.array(df['dp']*1e5/(df['rho']*9.81*7)) # pressure head [m]
# parameter sigmoid function  a0 = h_max, a1 = h_max - h_min, a2 = slope
a = [h.min(), h[:-1].max() - h.min(), 1, 40]# a3 = water cut transition
# Getting sigmoid PR through optimization ----------------------------------
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
# Get Data of DF and analysis data for all vibration analysis
# ==========================================================================
print('█████████ Getting data of one curve for sensor analysis ██████████')
curve = curves[13];  print('the selected curve is ███ %s ███'%(curve))
df = DF[DF['curve']==curve]       # dataframe for the corresponding curve
print( label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean())) # data
[i1, i2] = [np.where(df.index==df[i])[0][0] for i in ['i1', 'i2']]  #
t = h5todict('data/t.h5')['t']    # time vector 
vib = h5todict('data/' + curve + '.h5')  # vib dictionary in time
for key in keys:                         # get array of fft vibration
  signal = time_signal(vib[key], t, df['w'], 7)
  vib_arr = np.dstack((vib_arr, signal.Y)) if key!=keys[0] else signal.Y
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# Get fft plots all signals
# ==========================================================================
print('██████████████████████ Analysis of sensors ███████████████████████')
print('the selected curve is ███ %s ███'%(curve))
n = np.where(signal.f>=1000)[0][0]; x = df['wc']
y = signal.f[:n:512];               z = np.log10(vib_arr[:, :n:512, :])
lims = [0, i1 + 1, i2, len(x) + 1]
fig1 = scatter3d(x, y, z[:,:, 3:7], keys[3:7], lims)
fig2 = scatter3d(x, y, np.dstack((z[:,:, :3], z[:,:, 7:])), 
  keys[:3] + keys[7:], lims, cols=3)
#tikz_save('images/' + key + '.tex')
time_elapsed(START)             # Time elapsed in the process



# %% =======================================================================
# Spectrum fft analysis
# ==========================================================================
print('████████ Analysis of spectrum changes in phase inversion █████████')
print('the selected curve is ███ %s ███'%(curve))
for j, key in enumerate(keys):
  #if key == 'AC-05' or key=='AC-01Z': 
  n = np.where(signal.f>=750)[0][0]
  fig1 = spectrum(signal.f, vib_arr, df, [15, 30, 50, 57], j, n, keys)
  n1 = np.where(signal.f>=1000)[0][0];    n2 = np.where(signal.f>=5000)[0][0]  
  fig2 = spectrum(signal.f, vib_arr, df, [15, 30, 50, 57], j, n2, keys, n1=n1)
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# Parameter Results
# ==========================================================================
print('███████ Join statistics results for all data and analysis ████████')
key = keys[3];  
for key in keys:
  path_pr = '/'.join(['results', key + '_' + norm + '.h5'])   
  PR_i = pd.DataFrame(h5todict(path_pr))          # PR results
  PR_i = PR_i[columns]
  PR_i.columns =[ i + '_' + key for i in PR_i.columns]
  PR = PR_i if key== keys[0] else pd.concat([PR , PR_i], axis=1) 
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Results dataframe
# ==========================================================================
print('████ feature selection through pearson correlation analysis ██████')
fig1, features = pearson_bar(DF, PR, 'Ta', 'dimensionaless torque')
fig2, _ = pearson_bar(DF, PR, 'wc', 'water cut')
fig3, _ = pearson_bar(DF, PR, 'dp', '$\Delta p$ [bar]')
fig4, _ = pearson_bar(DF, PR, 'Ta', 'dimensionaless torque', 
  ini='fft_rms', n=5)
fig5, _ = pearson_bar(DF, PR, 'Ta', 'dimensionaless torque', 
  end='AC-05', n=5)
fig6, _ = pearson_bar(DF, PR, 'Ta', 'dimensionaless torque', 
  end='AC-01Y', n=5)
feat_s = ['_'.join(f.split('_')[:-1]) for f in features];
keys_s = [f.split('_')[-1] for f in features]; 
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Evaluate features
# ==========================================================================
print('██████████████ feature comparison and evaluation █████████████████')
# sort curves by rpm, flow -------------------------------------------------
curves = list(DF['curve'].unique());  Q = []
for curve in curves:
  df = DF[DF['curve']==curve]; Q.append(df['Q'].mean())
curves = [curves[i] for i in np.argsort(Q)]
curves.remove('s_2ph_32c_2400rpm_221m3h')
curves = curves[:10] + ['s_2ph_32c_2400rpm_221m3h'] + curves[10:]
# plot best features -------------------------------------------------------
for feat, key in zip(feat_s, keys_s):
  path_pr = '/'.join(['results', key + '_' + norm + '.h5'])   # path   
  PR = pd.DataFrame(h5todict(path_pr))                        # parameters
  scatter(DF, PR, curves, feat, key)
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# Logistic regression
# ==========================================================================
print('█████████████████████ logistic regression ████████████████████████')
pred = lambda x, threshold: 1*(x>threshold)
score = lambda y, y_pred: sum((y_pred==y)*1)/len(y)
lgc = ['x', 'y', 'train', 'w', 'threshold']
for feat, key in zip(feat_s, keys_s):
#feat = 'fft_rms_0_2hHz'                        
#for key in keys:
  LG = pd.DataFrame(columns=lgc); bs = []
  path_pr = '/'.join(['results', key + '_' + norm + '.h5'])   
  PR = pd.DataFrame(h5todict(path_pr))          # PR results
  for w in [30, 40, 50]:
    I = DF[np.round(DF['w']/60)==w][DF['class']!=0].index  # index
    # initial values for logit function ------------------------------------
    rng = default_rng(23)   # random number
    n1 = rng.choice(len(I), size=int(len(I)*0.25), replace=False) # training 
    n2 = np.array([i for i in range(len(I)) if all(i!=n1)])     # test index
    x = np.abs(np.array(PR[feat].loc[I]));   x_train = x[n1] # x train
    y = 2 - np.array(DF['class'].loc[I]); y_train = y[n1] # y train
    # initial values for logit function ------------------------------------
    b = [0, 0]; b[1], b[0], _, _, _ = linregress(x_train,y_train)
    # resolve logit function -----------------------------------------------
    b = minimize(lambda b: cost(b, x_train, y_train), b).x  # optimize
    # plot logit function --------------------------------------------------
    x_l = np.linspace(x.min(), x.max(), num=1000) # parameter
    p_l = logit(b, x_l)                 # likelihood through logit function
    threshold = x_l[np.where(p_l>=0.5)[0][0]] # division
    for n, c in zip([n1, n2], [0, 1]):
      ln = len(n)
      data = np.c_[x[n], y[n], ln*[c], ln*[w], ln*[threshold]]
      lg = pd.DataFrame(data, columns=lgc)
      LG = pd.concat([LG, lg] , axis=0, ignore_index=True)
    bs.append(b)
  plot_logit(LG, bs, key, feat)
  
time_elapsed(START)             # Time elapsed in the process






# %%



# %%

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
col1 = [  0,      0,       0, 1] # Color black            
col2 = [0.5, 0.3906, 0.24875, 1] # Color yellow
col3 = [0.3, 0.3000, 0.30000, 1] # Color gray
from functions_signal import get_var, time_signal, plot_lims
label1 = lambda w, mu, q : '%.0f [Hz], %.0f [mPa-s], %.0f \
  [$\mathrm{m^3/h}$]'%(w/60, mu, q)
def plot1(curves, DF,  x='wc', y='h', y_c=1e5/(9.81*7) ,
  xlabel='water cut %', ylabel='h [m]', loc='best', xlim=[0, 100]):
  '''simples dataframe plot by curves'''
  styles = ['o-.', 's--', 'd--', '.:']*10
  colors = [col1, col2, col2, col3]*10
  fig, ax = plt.subplots()
  for i, curve in enumerate(curves):
    df = DF[DF['curve']==curve] # used dataframe
    label = label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean())
    df_y = df['dp']/df['rho'] if y=='h' else df[y]
    ax.plot(df[x], df_y*y_c, styles[i], label=label, color=colors[i])
  ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel); 
  ax.set_xlim(xlim);      ax.legend(loc=loc)
  return fig
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
  gc = (.8,.8,.8,);   # Grid Color
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
    ax2.plot(f[n1:n2], fft[i, n1:n2, 1], alpha=0.6, color=col2)
    ax2.set_ylim([6e-5, 4e-1]); ax2.set_yticks([1e-4, 1e-3, 1e-2, 1e-1 ])
    # lin plot -------------------------------------------------------------
    ax.plot(f[n1:n2], fft[i, n1:n2, j], label=('wc = %.2f%%'%wc),color=col1)
    ax.set_ylim([0, m*9/8]);  ax.set_xlim([f[n1], f[n2]]);  
    ax.legend(loc='upper left')
    ax.set_ylabel('a [$\mathrm{m/s^2}$]', color=col1)
    ax2.set_ylabel('a [$\mathrm{m/s^2}$]', color=col2)
    ax.set_zorder(ax2.get_zorder()+12) # set axis order 
    ax.patch.set_visible(False)        # set axis order 

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
  y1 = ax[0].barh(features, values, color=col1) # figure with barh
  ax[0].set_xlim([ values.min()*0.99, values.max()*1.01 ])     # x lims
  ax[0].bar_label(y1, color='k', fmt='%.2f')
  ax[0].set_xlabel('Pearson R Coefficient')
  # get Data Frame ---------------------------------------------------------
  ax[1].plot(y, X[features[-1]], '.', color=col2)  # plot best feature
  ax[1].set_xlabel(ylabel); ax[1].set_ylabel(features[-1]);   # format
  return fig, list(features), list(values)
def logit(b, x):
  b = b.reshape(len(b), 1)
  return 1/ (1 + np.exp( -(b[0, 0] + x @ b[1:,:]  ) ))
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
  colors = [col3, col2, col1] # colors for 
  for i, m in enumerate(['_', 'o', 's']):   # for every class  
    I = np.where(z==i)[0]                   # index
    ax.plot(x[I], y[I], m, color=colors[i]) # plot 
  lb, ub = PR[pr_i].min(), PR[pr_i].max(); ax.set_xlim(lb, ub)
  ax.hlines([9.5, 13.5], lb, ub, color=(.0,.0,.0), linewidth=0.3)
  ax.set_xlabel(pr_i);                ax.set_title(key)
  ax.set_yticks(range(len(curves)));  ax.set_yticklabels(labels)
  ax.legend(['transition', 'water-in-oil', 'oil-in-water'])
  return fig
def plot_logit(LG, bs, key, feature, scores):
  fig, axs = plt.subplots(3, 1, sharex=True);
  fig.suptitle(key,x=0.2, y=0.93, fontsize=14)
  for ax, w, b, in zip(axs, [30, 40, 50], bs):
    # plot training and test data ------------------------------------------
    lg = LG[np.round(LG['w'])==w];        t = lg['threshold'].mean()
    train =lg[lg['train'] == 1];  test = lg[lg['train'] == 0]
    ax.plot(test['x'], test['y'], 'o', color=col2, 
      markersize=7, alpha=0.5)  # test data
    ax.plot(train['x'], train['y'], 'x', color=col1, markersize=10,
      markeredgewidth=2)  
    ax.set_xlim([LG['x'].min(), LG['x'].max()]);  ax.set_ylim([-0.05, 1.05])
    # plot logit function --------------------------------------------------
    x_l = np.linspace(LG['x'].min(), LG['x'].max(), num=1000) # parameter
    p_l = logit(b, np.array([x_l]).T)   # likelihood through logit function
    ax.plot(x_l, p_l, '--', color=col3) # plot logit
    ax.vlines(t,-0.1, 1.1, color=col1, linestyles=':')
    ax.set_ylabel('class at %s [Hz]'%w);  
    ax.set_yticks([0, 0.5, 1]); ax.grid('on'); 
  fig.text(0.25, 0.9, 'train score =  %.4f\ntest score = %.4f '%( 
    scores[1], scores[0]), fontsize=12)
  ax.set_xlabel(feature, fontsize=12); fig.subplots_adjust(hspace=0)
  fig.legend(['test', 'train', 'logit function'], 
    fontsize=12, bbox_to_anchor=(0.9, 0.95), ncol=3)
  return fig
pred = lambda x, threshold: 1*(x>threshold)
score = lambda y, y_pred: sum((y_pred==y)*1)/len(y)
def train_test(size, train_percentage, random_number=23):
  rng = default_rng(random_number)   # random number
  n_train = rng.choice(
    size, size=int(size*train_percentage/100), replace=False)     
  n_test = np.array([i for i in range(size) if all(i!=n_train)])
  return n_train, n_test
def threshold(x, b):
  x_l = np.linspace(x[:,0].min(), x[:,0].max(), num=10000) # parameter
  p_l = logit(b, np.array([x_l]).T) # likelihood through logit function
  return x_l[np.where(p_l>=0.5)[0][0]] # division

print('Function created  ████████████████████████████████████████████████')

# %% =======================================================================
# Get Data of DF and analysis data
# ==========================================================================
print('██████████████████████████████ Keys ██████████████████████████████')
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
# sort curves by rpm, flow -------------------------------------------------
print('██████████████████████████████ Curves █████████████████████████████')
curves = list(DF['curve'].unique());  Q = []; curve_a = curves[5]
for curve in curves:
  df = DF[DF['curve']==curve]; Q.append(df['Q'].mean())
curves = [curves[i] for i in np.argsort(Q)];  curves.remove(curve_a)
curves = curves[:10] + [curve_a] + curves[10:]; enum_vec(curves)
print('Analyses data defined...')
time_elapsed(START)                           # Time elapsed in the process

# %% =======================================================================
# Initial Curves
# ==========================================================================
print('█████████████████████ wc vs h, and wc vs Ta ██████████████████████')
curves_sel = [curves[i] for i in [16, 11, 12, 6]]
# Curve  of h vs wc --------------------------------------------------------
fig = plot1(curves_sel, DF, loc='upper left');  plt.ylim([2,12]); plt.show()
tikz_save('images/H.tex', figure=fig)
# Curve  of Ta vs wc -------------------------------------------------------
fig = plot1(curves_sel, DF, y='Ta', y_c=1/7, ylabel='dimesionaless torque', 
  loc='upper right');   plt.show()
tikz_save('images/Ta.tex', figure=fig)
time_elapsed(START)             # Time elapsed in the process

#%% ========================================================================
# Sigmoid functions
# ==========================================================================
# Getting  bound between water-in-oil & transition & oil-in-water ----------
print('███████████████████ Genering sigmoid function ████████████████████')
df = DF[DF['curve']==curves[12]] # dataframe for only this curve
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
limit = np.where(dsigmoid(a, x)>dy_max*0.2)[0]  # transition region
l1 = np.where(wc<x[ limit[ 0] ])[0][-1]         # begin trans
l2 = np.where(wc>x[ limit[-1] ])[0][ 0]         # end   trans
plot_sigmoid(x, a, wc, h, l1, l2, a[3]) 
plt.title(label1(df['w'].mean(), df['mu'].mean(), df['Q'].mean()))
time_elapsed(START)             # Time elapsed in the process


# %% =======================================================================
# Get Data of DF and analysis data for all vibration analysis
# ==========================================================================
print('█████████ Getting data of one curve for sensor analysis ██████████')
curve = curves[2];  print('the selected curve is ███ %s ███'%(curve))
df = DF[DF['curve']==curve]       # dataframe for the corresponding curve
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
index = [15, 30, 50, 57];
[n, n1, n2] = [np.where(signal.f>=i)[0][0] for i in [750, 1000, 5000]]
for j, key in enumerate(keys):                # AC-05 AC-01Z
  if key == 'AC-05' or key == 'AC-01Z':       # 
    fig1 = spectrum(signal.f, vib_arr, df, index, j, n, keys)
    fig2 = spectrum(signal.f, vib_arr, df, index, j, n2, keys, n1=n1)
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Parameter Results
# ==========================================================================
print('███████ Join statistics results for all data and analysis ████████') 
for key in keys:
  path_pr = '/'.join(['results', key + '_' + norm + '.h5'])   
  PR_i = pd.DataFrame(h5todict(path_pr))    # PR results
  PR_i = PR_i[columns]                      # filter by columns
  PR_i.columns =[ i + '_' + key for i in PR_i.columns] # add key name
  PR = PR_i if key== keys[0] else pd.concat([PR , PR_i], axis=1) 
time_elapsed(START)             # Time elapsed in the process
# Results dataframe --------------------------------------------------------
print('████ feature selection through pearson correlation analysis █████')
#fig1, _, _ = pearson_bar(DF, PR, 'wc', 'water cut')
#fig2, _, _ = pearson_bar(DF, PR, 'dp', '$\Delta p$ [bar]')
FT = pd.DataFrame(columns=keys, index=range(4)) # Resume r value for keys
for key in keys:
  _ , f, v = pearson_bar(DF, PR, 'Ta', '',  end=key, n=4); plt.close(); 
  f = [i[:-len(key) - 1] for i in f]      # features and values
  FT.index = f if key == keys[0] else FT.index; FT.loc[f, key] = v
fig3, features, _ = pearson_bar(DF, PR,   # figure of person parameters
  'Ta', 'dimensionaless torque', ini='fft_rms', n=9)
F = ['_'.join(f.split('_')[:-1]) for f in features] # Features to analyse
K = [f.split('_')[-1] for f in features]            # Keys to analyse
time_elapsed(START)                       # Time elapsed in the process
FT = FT.sort_values('AC-06Y', axis=0, ascending=False); FT

# %% =======================================================================
# compare features behavior
# ==========================================================================
print('██████████████████ feature behavior comparison ███████████████████')
# plot best features -------------------------------------------------------
for feature, key in zip(F[-2:], K[-2:]):    # evaluate the two best features
  path_pr = '/'.join(['results', key + '_' + norm + '.h5'])   # path   
  PR = pd.DataFrame(h5todict(path_pr))                        # parameters
  scatter(DF, PR, curves, feature, key)     # plot simple comparison
time_elapsed(START)             # Time elapsed in the process

# %% =======================================================================
# Logistic regression two dimension
# ==========================================================================
print('█████████████████████ logistic regression ████████████████████████')
I = DF[DF['class']!=0].index; m = len(I) # filter class 1 and 2
train, test = train_test(m, 10)   # get aleatory train division of 10%
split = np.zeros(m);  split[train] = 1  # data is for train=1 or test=0
y = 2 - np.array(DF['class'].loc[I]); y = y.reshape(m, 1)   # y
ws = [30, 40, 50]                 # angular velocities in Hz
FT = pd.DataFrame(index=['test score', 'train score']) # Result dataframe
feature = F[0]; bds = {}          # 'fft_rms_0_2hHz' , boundaries
print('the selected feature is ███ %s ███'%(feature))
for key in keys:                  # Evaluate all sensors
  PR = pd.DataFrame(h5todict('results/' + key + '_' + norm + '.h5'))    
  X = np.c_[np.abs(PR[feature].loc[I]) , DF['w'].loc[I]/60] # X = [f, w]  
  # resolve logit function -----------------------------------------------
  opt = minimize(lambda b: cost(b, X[train], y[train]), [0,0,0])# optimize 
  b = opt.x       # predicted constants of logit function
  bs = [np.array([b[0] + b[2]*w, b[1]]) for w in ws]  # adjust to one dim
  bds[key] = [threshold(X, b) for b in bs]# predicted threshold boundaries
  bounds = np.sum([
    (np.round(X[:,1])==w)*t for w, t in zip(ws, bds[key] )], axis=0)
  # plot logit function --------------------------------------------------
  LG = pd.DataFrame(np.c_[X, y[:,0], split, bounds], # Logistic Regresion
    columns= ['x', 'w', 'y', 'train','threshold'])
  FT[key] = [score(y[n,0], pred(X[n, 0], bounds[n])) for n in [test, train]]
  if any(key == np.array(['AC-01Y', 'AC-04', 'AC-06Y'])):
    plot_logit(LG, bs, key, feature, FT[key])
time_elapsed(START);             # Time elapsed in the process
FT.sort_values('test score', axis=1, ascending=False)

# %% =======================================================================
# Plot line regression comparison
# ==========================================================================
styles = ['o--', 'd--', 's--', 'x-', '+-', '.-', '|--', 'o:', 'd:', 's:']
colors = [col1, col2, col3]
for i, key in enumerate(bds):
  c = colors[1] if len(key)==5 else colors[0]
  c = colors[2] if key[4]=='6' else c
  ax =plt.plot(bds[key], ws, styles[i], label=key, color=c)
plt.legend(); plt.xlabel('$\Omega $ [Hz]'); plt.ylabel('fft rms 0-250 [Hz]')

# %% =======================================================================
# plot final result
# ==========================================================================
key = 'AC-04'; print('the selected key is ███ %s ███'%(key))
PR = pd.DataFrame(h5todict('results/' + key + '_' + norm + '.h5'))    
X = np.c_[np.abs(PR[feature].loc[I]) , DF['w'].loc[I]/60] # X = [f, w]
bounds = np.sum([
  (np.round(X[:,1])==w)*t for w, t in zip(ws, bds[key] )], axis=0)
LG = pd.DataFrame(np.c_[X, y[:,0], split, bounds], # Logistic Regresion
    columns= ['x', 'w', 'y', 'train','threshold'])
LG['Q'] = np.array(DF.loc[I]['Q'])
# linear regression for plane ----------------------------------------------
regr =linregress(ws, bds[key]); o = np.linspace(29,51)  # Omega Z
q = np.linspace(LG['Q'].min()*0.9, LG['Q'].max()*1.1)   # flow Y
O, Q  = np.meshgrid(o, q);  m = [".", "x"]  # Meshgrip m
fig = plt.figure(figsize=(13, 5)); 
for i, title in enumerate(["test", "train"]):    # for in 0 test, 1 train
  df1 = LG[LG['train']==i]              # filter by train test
  ax = fig.add_subplot(121 + i, projection='3d') # Add figure
  ax.set_box_aspect(aspect = (6,4,2))   # Axis aspect ration
  Pr = regr.slope*O  + regr.intercept   # Parameter
  ax.plot_wireframe(Pr, Q, O,  alpha=0.5, color=col3) #Plot boundary
  for j, col in enumerate([col1, col2]): # 0 water-in-oil, 1 oil-in-water
    df = df1[df1['y']==j]                     # filter by class
    ax.scatter3D(df['x'], df['Q'], df['w'],   # plot
      color=col, marker=m[j], alpha=1, s=50)
  ax.set_xlabel(feature);         ax.set_ylabel('Q [$\mathrm{m^3/h}$]')    
  ax.set_zlabel('$\Omega$ [Hz]'); ax.set_xlim(LG['x'].min(), LG['x'].max())
  ax.set_zlim(30,50);             ax.set_ylim(LG['Q'].min(), LG['Q'].max()) 
  ax.view_init(22, 250);          ax.set_title(title)
fig.legend(["boundary", "oil-in-water", "water-in-oil"]); fig.tight_layout()


# %%

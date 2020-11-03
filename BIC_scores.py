import numpy as np
import matplotlib.pyplot as plt
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse

# Data taken from https://www.atnf.csiro.au/research/pulsar/psrcat/
data = np.loadtxt("pulsar_data.txt",dtype = np.float64)
p_in = data[:,1]
p_err_in = data[:,2]
p_dot_in = data[:,3]
p_dot_err_in = data[:,4]
p_true=[]
p_err=[]
p_dot_true=[]
p_dot_err=[]

for i in range(np.size(p_err_in)):
    if p_dot_in[i]>0 and p_dot_in[i]!=9 and p_err_in[i]!=0:  #42 p_dot values are negative. Can't do scatter plot with negative values.
        p_true.append(p_in[i])
        p_err.append(p_err_in[i])
        p_dot_true.append(p_dot_in[i])
        p_dot_err.append(p_dot_err_in[i])
   
p_true=np.log(p_true)
p_dot_true=np.log(p_dot_true)
p_err=p_err/p_true
p_dot_err=p_dot_err/p_dot_true
X = np.vstack([p_true, p_dot_true]).T
Xerr = np.zeros(X.shape + X.shape[-1:])
diag = np.arange(X.shape[-1])
Xerr[:, diag, diag] = np.vstack([p_err**2, p_dot_err**2]).T
xd = XDGMM(6,200)

param_range = np.array([1,2,3,4,5,6,7,8,9,10])
bic,optimal_n_comp,lowest_bic = xd.bic_test(X,Xerr,param_range)
plt.figure()
ax=plt.axes([0.17,0.17,0.8,1])
plt.plot(param_range,bic,'o-', color='blue', label='BIC Score', linewidth=1.75, markersize=12)
plt.xlabel('Number of Components', fontsize='18')
plt.ylabel('BIC score', fontsize='18')
plt.legend(prop={'size':22})
plt.style.use('seaborn-whitegrid')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=0)
plt.show()

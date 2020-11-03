#XDGMM implementation submit
import numpy as np
import matplotlib.pyplot as plt
from xdgmm import XDGMM
from astroML.plotting.tools import draw_ellipse
import time
start_time = time.time()

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
# p_err=np.log(p_err)
# p_dot_err=np.log(p_dot_err)

p_err=p_err/p_true
p_dot_err=p_dot_err/p_dot_true


X = np.vstack([p_true, p_dot_true]).T
Xerr = np.zeros(X.shape + X.shape[-1:])
diag = np.arange(X.shape[-1])
Xerr[:, diag, diag] = np.vstack([p_err**2, p_dot_err**2]).T

xd = XDGMM(6,200)
# xd.fit(X, Xerr) 

#------------------------------------------------------------------------------------------Plot of Original data points


# ax1 = fig.add_subplot(411)
# ax1.scatter(p_true,p_dot_true, marker='o',color=(0,0.2,1), alpha=1, s=[8])
# ax1.set_xlabel('PERIOD [s]')
# ax1.set_ylabel('PERIOD DERIVATIVE [s/s]')

# ax1.scatter(np.exp(p_true),np.exp(p_dot_true), marker='.')
# ax1.set_xscale('log')
# ax1.set_yscale('log')


# plt.scatter(p_true,p_dot_true, marker='o',color=(0,0.2,1), alpha=1, s=[8])
# plt.xlabel('PERIOD [s]', fontsize='16')
# plt.ylabel('PERIOD DERIVATIVE [s/s]', fontsize='16')
# plt.style.use('seaborn-whitegrid')


# plt.scatter(np.exp(p_true),np.exp(p_dot_true), marker='.',color=(0,0.2,1), alpha=1, s=[14])
# plt.xscale('log')
# plt.yscale('log')

#------------------------------------------------------------------------------------------Plot of Sampled data points

# np.random.seed(42)
# sample = xd.sample(2000)

# ax2= fig.add_subplot(412)
# ax2.scatter(sample[:, 0], sample[:, 1], s=4, lw=0, c='k')
# ax2.set_xlabel('P(s)')
# ax2.set_ylabel('P_dot(s/s)')
# ax2.title.set_text('Sampled data')

# ax2.scatter(np.exp(sample[:, 0]), np.exp(sample[:, 1]), s=4, lw=0, c='k')
# ax2.set_xscale('log')
# ax2.set_yscale('log')

# #---------------------------------------------------------------------------------------------------Plot of BIC scores

param_range = np.array([1,2,3,4,5,6,7,8,9,10])
bic,optimal_n_comp,lowest_bic = xd.bic_test(X,Xerr,param_range)

# ax3 = fig.add_subplot(413)
# ax3.plot(param_range,bic)
# ax3.set_xlabel('Number of Components')
# ax3.set_ylabel('BIC score')
# ax3.title.set_text('BIC plot')

plt.figure()
ax=plt.axes([0.17,0.17,0.8,1])

plt.plot(param_range,bic,'o-', color='blue', label='BIC Score', linewidth=1.75, markersize=12)
plt.xlabel('Number of Components', fontsize='18')
plt.ylabel('BIC score', fontsize='18')
plt.legend(prop={'size':22})
plt.style.use('seaborn-whitegrid')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=0)

# #---------------------------------------------------------------------------------------------------Plot of Ellipses

# for i in range(xd.n_components):
#     draw_ellipse((xd.mu[i]), xd.V[i], scales=[2], ax=ax1, ec='k', fc='red', edgecolor='purple',linestyle='--', alpha=0.3)

plt.savefig('bic_score.png',bbox_inches='tight')
plt.savefig('bic_score.pdf',bbox_inches='tight')
    
print("%s seconds" % (time.time() - start_time))
plt.show()
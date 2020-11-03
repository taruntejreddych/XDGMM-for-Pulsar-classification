import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astroML.density_estimation import XDGMM
# matplotlib.rcParams['text.usetex'] = False
from xdgmm import XDGMM
# print(np.log(2))
# Data taken from https://www.atnf.csiro.au/research/pulsar/psrcat/
data = np.loadtxt("pulsar_data.txt",dtype = np.float64)
p_in = data[:,1]
p_err_in = data[:,2]
p_dot_in = data[:,3]
p_dot_err_in = data[:,4]


#Data after filtering out unknown values negative P derivatives(for doing a scatter plot)
p_true=[]
p_err=[]
p_dot_true=[]
p_dot_err=[]

for i in range(np.size(p_err_in)):
    if p_dot_in[i]>0 and p_dot_in[i]!=9:  #42 p_dot values are negative. Can't do scatter plot with negative values.
        p_true.append(p_in[i])
        p_err.append(p_err_in[i])
        p_dot_true.append(p_dot_in[i])
        p_dot_err.append(p_dot_err_in[i])
print(np.size(p_true))
plt.figure()
ax=plt.axes([0.17,0.17,0.8,1])


plt.scatter(p_true,p_dot_true, marker='o',color=(0,0.05,1), alpha=1, s=[8])
# plt.errorbar(p_true, p_dot_true, xerr=p_err, yerr=p_dot_err,color=(0,0.3,1),alpha=0.7, linestyle="None",ecolor='red',
#              elinewidth=1, capsize=1.2)


plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-3,5e1)
plt.ylim(1e-22,1e-9)

plt.xlabel('PERIOD [s]', fontsize='18',fontweight='bold')
plt.ylabel('PERIOD DERIVATIVE [s/s]', fontsize='18',fontweight='bold')
plt.style.use('seaborn-whitegrid')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=0)


# plt.savefig('P_Pdot_plot.png',bbox_inches='tight')
# plt.savefig('P_Pdot_plot.pdf',bbox_inches='tight')
# plt.savefig('P_Pdot_plot_error.png',bbox_inches='tight')
# plt.savefig('P_Pdot_plot_error.pdf',bbox_inches='tight')

plt.show()
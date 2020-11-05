from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
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
xd.fit(X, Xerr) 

def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)                #the radius are multiplied by 2, which makes the Confidence Intervals as 68%.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
   
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

fig=plt.figure()
ax1=plt.axes([0.17,0.17,0.9,1.1])
ax1.scatter(p_true,p_dot_true, marker='o',color=(0,0.2,1), alpha=1, s=[8])
ax1.set_xlabel('log(PERIOD) [s]', fontsize='18')
ax1.set_ylabel('log(PERIOD DERIVATIVE) [s/s]', fontsize='18')
plt.style.use('seaborn-whitegrid')
color=['green','red','purple','magenta','black','orange']
for i in range(xd.n_components):
    confidence_ellipse(xd.mu[i][0], xd.mu[i][1], xd.V[i], ax1, linewidth=2, edgecolor=color[i])
    print(xd.weights[i],xd.mu[i][0],xd.mu[i][1], xd.V[i])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=0)
plt.show()

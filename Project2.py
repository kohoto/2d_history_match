import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math as m
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

# Define normal dist.
normal = lambda x, sigma, mu: 1/(sigma*m.sqrt(2*m.pi)) * np.exp(-0.5*((x-mu)/sigma)*((x-mu)/sigma))

#===============
#
#  Part 1
#
#===============
true=[]
with open('Truemodel.txt') as f:
    for line in f.readlines():
        true.append([float(x) for x in line.replace('\n', '').split('\t')])
true = np.array(true)

noisy = []
with open('Noisydata.txt') as f:
    for line in f.readlines():
        noisy.append([float(x) for x in line.replace('\n', '').split('\t')])
noisy = np.array(noisy)

wavelet = []
with open('Wavelet.txt') as f:
    for line in f.readlines():
        wavelet.append([float(x) for x in line.replace('\n', '').split('\t')])
wavelet = np.array(wavelet)
wavelet2 = []
with open('Wavelet2.txt') as f:
    for line in f.readlines():
        wavelet2.append([float(x) for x in line.replace('\n', '').split('\t')])
wavelet2 = np.array(wavelet)

# try my wavelet
f = interp1d(wavelet2[:,0], wavelet2[:,1]) # , kind='cubic')
wavelet[:,1] = f(wavelet[:,0])

amp = []
with open('Ampspec.txt') as f:
    for line in f.readlines():
        amp.append([float(x) for x in line.replace('\n', '').split('\t')])
amp = np.array(amp)

# noise free data
n_par = 2
n_data = noisy.shape[0]
Z = 5e6
dZ_true = 3e6
dt_true = 1.5
t1 = 50
ts = noisy[:,0]

# Define wavelet interpolation function.
wavelet_interp = interp1d(wavelet[:,0], wavelet[:,1], kind='cubic')
# Use this when you interpolate the data values. since it's
wavelet_interp_l = interp1d(wavelet[:,0], wavelet[:,1])

s = np.array(dZ_true / (2*Z+dZ_true) * (wavelet_interp(ts-t1) - wavelet_interp(ts-t1-dt_true))).transpose()

# noise
noise = noisy[:,1] - s

# Recreate Figure2
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(true[:,0], true[:,1])
axs[0, 0].set_xlabel('time [ms]')
axs[0, 0].set_ylabel('impedance')
axs[1, 0].plot(wavelet[:,0], wavelet[:,1])
axs[1, 0].set_xlabel('time [ms]')
axs[1, 0].set_ylabel('wavelet')
axs[2, 0].plot(amp[:,0], amp[:,1])
axs[2, 0].set_xlabel('time [ms]')
axs[2, 0].set_ylabel('amplitude spectrum wavelet [dB]')
axs[0, 1].plot(noisy[:,0], s)
axs[0, 1].set_xlabel('time [ms]')
axs[0, 1].set_ylabel('noise free')
axs[0, 1].set_ylim((-2.5, 2.5))
axs[0, 1].grid()
axs[1, 1].plot(noisy[:,0], noise)
axs[1, 1].set_xlabel('time [ms]')
axs[1, 1].set_ylabel('noise')
axs[1, 1].set_ylim((-2.5, 2.5))
axs[1, 1].grid()
axs[2, 1].plot(noisy[:,0], noisy[:,1])
axs[2, 1].set_xlabel('time [ms]')
axs[2, 1].set_ylabel('noisy data')
axs[2, 1].set_ylim((-2.5, 2.5))
axs[2, 1].grid()
fig.set_size_inches(8.5, 11) # letter width
plt.savefig('part1_fig2.png')

# priori distribution
# dZ
mu_pri = np.array([3.4e6, 3])
sigma_pri = np.array([0.5e6, 2])
dZ = np.arange(1,6,0.1)*1e6
dZ_priori = normal(dZ, sigma_pri[0], mu_pri[0])
# dt
dt = np.arange(-3,9,0.1)
dt_priori = normal(dt, sigma_pri[1], mu_pri[1])
dt_priori[dt<=0] = 0

# likelihood
mu_n = sum(noisy[:,1]) / noisy[:,1].shape[0]
noisy_std = np.sqrt(sum((noisy[:,1] - mu_n)**2) / noisy[:,1].shape[0])
# Q. not sure if I should use noise or noisy data to calc s.t.d?
sigma_n = np.std(noise)
# sigma_n = np.std(noisy[:,1])

w = np.interp(ts-t1, wavelet[:,0], wavelet[:,1])
i = 0
p_data = np.zeros((dt.size,dZ.size))
for dt1 in dt:
    j = 0
    for dZ1 in dZ:
        w2 = np.interp(ts-t1-dt1, wavelet[:,0], wavelet[:,1])
        s = dZ1 / (2*Z+dZ1) * (w - w2)
        p_data[i,j] = np.exp(-0.5/sigma_n**2*np.sum((noisy[:,1]-s)*(noisy[:,1]-s)))
        j += 1
    i += 1

#===============
#  Fig. 3
#===============
fig, axs = plt.subplots(1, 2)
axs[0].plot(dZ, dZ_priori)
axs[0].set_xlabel('dZ')
axs[0].set_ylabel('P(dZ)')
axs[1].plot(dt, dt_priori)
axs[1].set_xlabel('dt')
axs[1].set_ylabel('P(dt)')
fig.set_size_inches(8.5, 4) # letter width
plt.savefig('part1_fig3.png')


# priori distribution
p_prior = np.tile(dZ_priori, (dt.size, 1)) *  np.tile(dt_priori, (dZ.size, 1)).transpose()
dZ_2d = np.tile(dZ, (dt.size, 1))
dt_2d = np.transpose(np.tile(dt, (dZ_2d.shape[1], 1)))

# post distribution
p_post = p_prior*p_data

#===============
#  Fig. 4
#===============
aspect = (4, 4, 1)
fig = plt.figure()
# set up the axes for the first plot
ax = fig.add_subplot(3, 1, 1, projection='3d')
ax.plot_surface(dZ_2d, dt_2d, p_prior, cmap=cm.coolwarm)
ax.set_xlabel("dZ")
ax.set_ylabel("dt")
ax.set_xlim3d((1e6, 6e6))
ax.set_ylim3d((-4, 6))
ax.set_box_aspect(aspect)
ax.patch.set_alpha(0.)
ax.zaxis.set_major_locator(plt.NullLocator()) # delete tick marks on z-axis
ax.view_init(30, -30)
# ax.set_zlim3d(XYZlim)
ax.invert_xaxis()

# set up the axes for the likelihood plot
ax = fig.add_subplot(3,1, 2, projection='3d')
ax.plot_surface(dZ_2d, dt_2d, p_data, cmap=cm.coolwarm)
ax.set_xlabel("dZ")
ax.set_ylabel("dt")
ax.set_xlim3d((1e6, 6e6))
ax.set_ylim3d((-4, 6))
ax.set_box_aspect(aspect)
ax.patch.set_alpha(0.) # backgrond color = transpalent
ax.zaxis.set_major_locator(plt.NullLocator()) # delete tick marks on z-axis
ax.view_init(30, -30)
ax.invert_xaxis()

# set up the axes for the Posterior plot
ax = fig.add_subplot(3,1, 3, projection='3d')
ax.plot_surface(dZ_2d, dt_2d, p_post, cmap=cm.coolwarm)
ax.set_xlabel("dZ")
ax.set_ylabel("dt")
ax.set_xlim3d((1e6, 6e6))
ax.set_ylim3d((-4, 6))
ax.set_box_aspect(aspect)
ax.patch.set_alpha(0.)
ax.zaxis.set_major_locator(plt.NullLocator()) # delete tick marks on z-axis
ax.view_init(30, -30)
ax.invert_xaxis()
plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=1, hspace=-0.3)
fig.set_size_inches(8.5, 11) # letter width
plt.savefig('part1_fig4.png')

#===============
#  Fig. 5 - Top view of countour plot.
#================
# set up the axes for the prior
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.contour(dt_2d, dZ_2d, p_prior)
ax.plot(dt_true, dZ_true, 'r*', markersize=5)
ax.set_xlabel("dt")
ax.set_ylabel("dZ")
ax.set_xlim(-4, 6)
ax.set_ylim(1e6, 6e6)

# set up the axes for the likelihood
ax = fig.add_subplot(2, 2, 2)
ax.contour(dt_2d, dZ_2d, p_data)
ax.plot(dt_true, dZ_true, 'r*', markersize=5)
ax.set_xlabel("dt")
ax.set_ylabel("dZ")
ax.set_xlim(-4, 6)
ax.set_ylim(1e6, 6e6)

# set up the axes for the prior
ax = fig.add_subplot(2, 2, 3)
ax.contour(dt_2d, dZ_2d, p_post)
ax.plot(dt_true, dZ_true, 'r*', markersize=5)
ax.set_xlabel("dt")
ax.set_ylabel("dZ")
ax.set_xlim(-4, 6)
ax.set_ylim(1e6, 6e6)
fig.set_size_inches(8.5, 6) # letter width
plt.savefig('part1_fig5.png')


# create table for summary
MLE_idx = np.where(p_data == np.amax(p_data))
MLE_dZ = dZ[MLE_idx[1]]
MLE_dt = dt[MLE_idx[0]]
MAP_idx = np.where(p_post == np.amax(p_post))
MAP_dZ = dZ[MLE_idx[1]]
MAP_dt = dt[MLE_idx[0]]
max_dZ_prior = dZ[np.where(dZ_priori == dZ_priori.max())] #, np.where(dt_priori[0,:] == dt_priori[0,:].max())]
max_dt_prior = np.where(dt_priori == dt_priori.max())
#summary_dt = abs(np.array(MAP_dt)-dt_true)
#summary_dZ = abs(np.array(summary[1])-dZ_true)

# Summary table here.

w2 = np.interp(ts - t1 - dt[MLE_idx[0]], wavelet[:, 0], wavelet[:, 1])
s = dZ[MLE_idx[1]] / (2 * Z + dZ[MLE_idx[1]]) * (w - w2)
L2_MLE = sum((noisy[:,1]-s)**2) # / sum(noisy[:,1]**2)

w2 = np.interp(ts - t1 - dt[MAP_idx[0]], wavelet[:, 0], wavelet[:, 1])
s = dZ[MAP_idx[1]] / (2 * Z + dZ[MAP_idx[1]]) * (w - w2)
# Something wrong with norm
L2_MAP = np.sqrt(sum((noisy[:,1]-s)**2)) #/ np.sqrt(sum(noisy[:,1]**2))

# Plot the comparison of noisy data, residual, and noise
#================
# Top view of countour plot.
#================
# set up the axes for the prior
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.plot(noisy[:,0], noisy[:,1])
ax.set_xlabel('time [ms]')
ax.set_ylabel('wavelet')
ax = fig.add_subplot(3, 1, 2)
ax.plot(ts, noisy[:,1]-s)
ax.set_xlabel('time [ms]')
ax.set_ylabel('residuals')
ax = fig.add_subplot(3, 1, 3)
ax.plot(ts, noise)
ax.set_xlabel('time [ms]')
ax.set_ylabel('noise')
plt.savefig('part1_fig6.png')


#================
#
# PART 2
#
#================
# The least-squares problem

# The prior Covariance Matrix
Cx = np.diag([sigma_pri[0]**2, sigma_pri[1]**2])

# Lambda functions for sensitivity calculation
epsilon_dt = 0.0001 # small increment
# dt_MAP = dt[MAP_idx[0], 0]
# dZ_MAP = dZ[0, MAP_idx[1]]
dZ_MAP = 3.32e6
dt_MAP = 1.25

# wavelet calculation for sensitivity
w = np.interp(ts-t1, wavelet[:,0], wavelet[:,1])
ts2 = ts-t1-dt_MAP
w2 = np.interp(ts2, wavelet[:,0], wavelet[:,1]) # something wrong????
w2_epsilon = np.interp(ts-t1-dt_MAP-epsilon_dt, wavelet[:,0], wavelet[:,1])

G = np.zeros((n_data, n_par))
# derivative by dZ
G[:,0] = 2*Z/((2*Z + dZ_MAP)**2) * (w - w2) # dZ will be value
# derivative by dt
G[:,1] = dZ_MAP / (2*Z+dZ_MAP) * (w2 - w2_epsilon) / epsilon_dt

Cn = np.diag([sigma_n**2] * n_data)
D = sqrtm(Cx)
a = np.linalg.inv(np.diag([sigma_n] * n_data))
G_h = np.linalg.inv(np.diag([sigma_n] * n_data)) @ G @ D # same
# for confirmation calcualte U, S, V from sensitivity
u_from_G_h, sq_s_from_G_h, vh_from_G_h = np.linalg.svd(G_h)

# check if I get Hy_h by calculating V * S * S * V'
diags = sq_s_from_G_h**2
Hy_h_check = vh_from_G_h.transpose() @ np.diag(sq_s_from_G_h) @ np.diag(sq_s_from_G_h) @ vh_from_G_h

Hy_h = G_h.transpose() @ G_h # Hy_h should be wrong since the Cx and Cx_h is wrong.
u, s, vh = np.linalg.svd(Hy_h) # this s is wrong!! # s = [8.138, 0.008]
vh = vh.transpose() # actual v
lambdas = (s + 1) # lambdas = [9.138, 1.008]

# Checking by values on paper
# vh = np.array([[0.0684, -0.9977], [0.9977, 0.0684]]) # multiplied to [dZ1, dZ2], [dt1, dt2]
lambdas = np.array([9.138, 1.008])
# e1 = np.linalg.norm(vh[:,0], ord=2)
# e2 = np.linalg.norm(vh[:,1], ord=2)
# v = np.array([vh[:,0]/e1, vh[:,1]/e2])
v = D @ vh
#================
# Top view of countour plot with eigen vectors.
#================
# get dZ and dt along the first eigen vector
dZ_e1 = (v[0,0] / v[1,0]) * (dt - dt_MAP) + dZ_MAP
dZ_e2 = (v[0,1] / v[1,1]) * (dt - dt_MAP) + dZ_MAP
# set up the axes for the prior
plt.figure()
plt.contour(dt_2d, dZ_2d, p_post)
# plot eigen vectors
plt.plot(dt, dZ_e1, 'b')
plt.plot(dt, dZ_e2, 'r')
plt.xlabel("dt")
plt.ylabel("dZ")
plt.xlim(-4, 6)
plt.ylim(1e6, 5e6)
plt.savefig('part2_fig1.png')
# pri
plt.figure()
plt.contour(dt_2d, dZ_2d, p_prior)
# plot eigen vectors
plt.plot(dt, dZ_e1, 'b')
plt.plot(dt, dZ_e2, 'r')
plt.xlabel("dt")
plt.ylabel("dZ")
plt.xlim(-4, 6)
plt.ylim(1e6, 5e6)
plt.savefig('part2_fig12.png')
plt.figure()
plt.contour(dt_2d, dZ_2d, p_data)
# plot eigen vectors
plt.plot(dt, dZ_e1, 'b')
plt.plot(dt, dZ_e2, 'r')
plt.xlabel("dt")
plt.ylabel("dZ")
plt.xlim(-4, 6)
plt.ylim(1e6, 5e6)
plt.savefig('part2_fig13.png')
#================
# Figure 2 - PDF along eigen vectors
#================
f = interp2d(dZ, dt, p_prior)
p_prior_e1 = np.diag(f(dZ_e1, dt))
p_prior_e2 = np.diag(np.fliplr(f(dZ_e2, dt)))
f = interp2d(dZ, dt, p_data)
p_data_e1 = np.diag(f(dZ_e1, dt))
p_data_e2 = np.diag(np.fliplr(f(dZ_e2, dt)))
f = interp2d(dZ, dt, p_post)
p_post_e1 = np.diag(f(dZ_e1, dt))
p_post_e2 = np.diag(np.fliplr(f(dZ_e2, dt)))

# plot figures
# set up the axes for the dZ
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
#ax.plot(dt, np.diag(p_prior_e1)  / np.diag(p_prior_e1).max(), label = 'prior')
ax.plot(dt, p_prior_e1 / p_prior_e1.max(), label = 'prior')
ax.plot(dt, p_data_e1 / p_data_e1.max(), label = 'data')
ax.plot(dt, p_post_e1 / p_post_e1.max(), label = 'posterior')
ax.set_xlabel('time [ms]')
ax.set_ylabel('p_like')
ax.set_xlim(dt_MAP-1, dt_MAP+1)
ax.legend()

# Second eigen vector
ax = fig.add_subplot(1, 2, 2)
ax.plot(dt, p_prior_e2 / p_prior_e2.max(), 'b-' , label = 'prior')
ax.plot(dt, p_data_e2 / p_data_e2.max(), 'r-' , label = 'data')
ax.plot(dt, p_post_e2 / p_post_e2.max(),'k:' , label = 'posterior')
ax.set_xlabel('time [ms]')
ax.set_ylabel('p_like')
ax.set_xlim(dt_MAP-0.5, dt_MAP+0.5)
ax.legend()
fig.set_size_inches(8.5, 4) # letter width
plt.savefig('part2_fig2.png')


#================
# Figure 3 - priori and posteriori pdfs for impedance dZ & time thickness dt
#================
# The posterior Covariance Matrix
Cx_post_h = np.linalg.inv(Hy_h + np.identity(n_par))
Cx_post_h = np.array([[0.98, -0.066], [-0.066, 0.017]])
# Correlation matrix
D_post = np.array([[1 / np.sqrt(Cx_post_h[0,0]), 0], [0, 1 / np.sqrt(Cx_post_h[1,1])]])
Cor = D_post @ Cx_post_h @ D_post
H = np.linalg.inv(D.transpose()) @ (Hy_h + np.identity(n_par)) @ np.linalg.inv(D)

Cx_post_in_SI = np.linalg.inv(H) * np.array([[1, 1e-3], [1e-3, 1e-6]])
sigma_post = np.sqrt(np.diag(Cx_post_in_SI)) * np.array([1, 1e3]) # calc post s.t.d with units [kg2/m2/s2] and [ms]
# calculate the average and s.t.d of posterior
mu_post = [sum(p_post.sum(axis=0) * dZ) / p_post.sum(), sum(p_post.sum(axis=1) * dt) / p_post.sum()]
p_dZ_post = normal(dZ, sigma_post[0], mu_post[0])
p_dt_post = normal(dt, sigma_post[1], mu_post[1])

# dimensional resolution matrix
G = np.sqrt(Cn) @ G_h @ np.sqrt(np.linalg.inv(Cx)) # sensitivity matrix with dimension
R = np.linalg.inv(H) @ G.transpose() @ np.linalg.inv(Cn) @ G
R = R * np.array([[1, 1e3], [1e-3, 1]]) # ==> OUTPUT
# normalized resolution matrix
R_h = np.identity(n_par) - Cx_post_h # ==> OUTPUT

# plot figures - assume posterior follows Gaussian distribution
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(dZ, dZ_priori / dZ_priori.max(),'k:', label = 'prior')
ax.plot(dZ, p_dZ_post / p_dZ_post.max(),'r-', label = 'posterior')
ax.set_xlabel('dZ [kg/m2/s]')
ax.set_ylabel('p_like')
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.plot(dt, dt_priori / dt_priori.max(),'k:', label = 'prior')
a = dt_priori.sum()
ax.plot(dt, p_dt_post / p_dt_post.max(),'r-', label = 'posterior')
ax.set_xlabel('time [ms]')
ax.set_ylabel('p_like')
ax.legend()
fig.set_size_inches(8.5, 4) # letter width
plt.savefig('part2_fig3.png')

#================
# Figure 4 - Uncertainty intervals
#================
# prepare variables in plot
dts_list = [[mu_pri[1] - sigma_pri[1], mu_pri[1], mu_pri[1] + sigma_pri[1]], # prior
            [mu_post[1] - sigma_post[1], mu_post[1], mu_post[1] + sigma_post[1]], # post
            [mu_post[1] - sigma_post[1], dt_true, mu_post[1] + sigma_post[1]]] # true vs post
dZs_list = [[mu_pri[0] - sigma_pri[0], mu_pri[0], mu_pri[0] + sigma_pri[0]], # prior
            [mu_post[0] - sigma_post[0], mu_post[0], mu_post[0] + sigma_post[0]], # post
            [mu_post[0] - sigma_post[0], dZ_true, mu_post[0] + sigma_post[0]]] # true vs post
fig_titles = ['a) prior', 'b) posterior', 'c) posterior bounds vs true']
line_styles = ['k:', 'k-', 'k:']
i = 0
fig, axs = plt.subplots(3, 1)
for (dts, dZs, fig_title) in zip(dts_list, dZs_list, fig_titles):
    for (dt_plot, dZ_plot, line_style) in zip(dts, dZs, line_styles):
        time = [0, t1, t1, t1+dt_plot, t1+dt_plot, 100]
        impedance = [Z, Z, Z+dZ_plot, Z+dZ_plot, Z, Z]
        axs[i].plot(time, impedance, line_style)
        axs[i].set_title(fig_title)
        axs[i].set_xlabel('time [ms]')
        axs[i].set_ylabel('impedance')
        axs[i].set_xlim(45, 60)
        axs[i].set_ylim(2e6, 10e6)
    i += 1 # for figure number

fig.tight_layout()
fig.set_size_inches(8.5, 5) # letter width
plt.savefig('part2_fig4.png')

# #================
# # Figure 5 - Linearlity Check
# #================
# define lambda function for approx. posterior pdf
p_post_approx = lambda dZ, dt: np.exp(-0.5 / sigma_post[0]**2 * (dZ - mu_post[0])**2 - 0.5 / sigma_post[1]**2 * (dt - mu_post[1])**2)
p_post_approx_e1 = p_post_approx(dZ_e1, dt)
p_post_approx_e2 = p_post_approx(dZ_e2, dt)
e1 = np.sqrt(dt**2 + dZ_e1**2)
e2 = np.sqrt(dt**2 + dZ_e2**2)
# plot figures
# set up the axes for the dZ
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(e1, p_post_approx_e1 / p_post_approx_e1.max(), 'r:', label = 'approxmation')
ax.plot(e1, p_post_e1 / p_post_e1.max(), 'k-', label = 'true')
ax.set_xlabel('eigenvector 1')
ax.set_ylabel('p_like')
ax.set_xlim(3.29e6, 3.34e6)
ax.legend()

# Second eigen vector
ax = fig.add_subplot(1, 2, 2)
ax.plot(e2, p_post_approx_e2 / p_post_approx_e2.max(), 'r:', label = 'approxmation')
ax.plot(e2, p_post_e2 / p_post_e2.max(), 'k-', label = 'true')
ax.set_xlabel('eigenvector 2')
ax.set_ylabel('p_like')
ax.set_xlim(0.1e7, 0.55e7)
ax.legend()
fig.set_size_inches(8.5, 4) # letter width
plt.savefig('part2_fig5.png')
#
#================
# Figure 6 - Standard deviations and marginal PDFs
#================
p_post_approx_2D = p_post_approx(dZ_2d, dt_2d)

# plot marginal PDFs for true and approximated prior / posterior
fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].plot(dZ, np.sum(p_post, axis=0) / np.sum(p_post, axis=0).max(), 'k-', label = 'true')
axs[0].plot(dZ, np.sum(p_post_approx_2D, axis=0) / np.sum(p_post_approx_2D, axis=0).max(), 'r:', label = 'approxmation')
axs[0].set_xlabel('dZ')
axs[0].set_ylabel('p_like')
axs[0].legend()
axs[1].plot(dt, np.sum(p_post, axis=1) / np.sum(p_post, axis=1).max(), 'k-', label = 'true')
axs[1].plot(dt, np.sum(p_post_approx_2D, axis=1) / np.sum(p_post_approx_2D, axis=1).max(), 'r:', label = 'approxmation')
axs[1].set_xlabel('dt')
axs[1].set_ylabel('p_like')
axs[1].legend()
fig.set_size_inches(8.5, 4) # letter width
plt.savefig('part2_fig6.png')

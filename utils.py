import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
from matplotlib.colors import ListedColormap

def chop(trajs): 
    '''
    Chop up trajectories to the same lengths as the shortest one 
    '''
    L = min(map(len, trajs))
    new_trajs = [] 
    for traj in trajs: 
        start = 0 
        while start + L <= len(traj): 
            new_trajs.append(traj[start:start+L])
            start += L 
    return new_trajs

def find_auto_corr(x, norm=True): 
    corr = np.correlate(x, x, mode='full')
    L = x.size
    n = np.arange(L, 0, -1)
    corr = corr[L-1:]/n
    if norm: 
        corr /= corr[0]
    return corr


def find_corr(x, y, norm=True):
    corr = np.correlate(x, y, mode='full')
    L = x.size
    n = np.concatenate([np.arange(1, L, 1), np.arange(L, 0, -1)])
    corr /= n
    if norm: 
        corr /= np.sqrt(np.mean(x**2))*np.sqrt(np.mean(y**2))
    return corr

def plot_corr(data, N, dt, norm=True, tex=False, colors=['copper', 'midnightblue'], n_trajs=8):     
    t = dt*np.arange(N)
    t1 = dt*np.arange(-N+1, N)
    L = len(data[0][0])

    continuous_cmap = plt.get_cmap(colors[0])
    colorlist = continuous_cmap(np.linspace(0.3, 0.8, n_trajs))
    discrete_cmap = ListedColormap(colorlist)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[3, 1, 2]) 
    
    theta_corrs = [] 
    p_corrs = [] 
    cross_corrs = [] 
    for (i, (theta, p)) in enumerate(data): 
        theta_corrs.append(find_auto_corr(theta, norm=norm))
        p_corrs.append(find_auto_corr(p, norm=norm))
        cross_corrs.append(find_corr(theta, p, norm=norm))
        if i < n_trajs:
            axes[0].plot(t, theta_corrs[-1][:N], alpha=0.6, c=discrete_cmap(i))
            axes[1].plot(t, p_corrs[-1][:N], alpha=0.6, c=discrete_cmap(i))
            axes[2].plot(t1, cross_corrs[-1][L-N+1:L+N], alpha=0.6, c=discrete_cmap(i))
        
    axes[0].plot(t, np.mean(theta_corrs, axis=0)[:N], '--', c=colors[1])
    axes[1].plot(t, np.mean(p_corrs, axis=0)[:N], '--', c=colors[1])
    axes[2].plot(t1, np.mean(cross_corrs, axis=0)[L-N+1:L+N], '--', c=colors[1])
    axes[0].set_xlim([0, 50])
    axes[0].set_ylim([-0.4, 1])
    axes[1].set_xlim([0, 10])
    axes[1].set_ylim([-0.2, 1])
    axes[2].set_xlim([-50, 50])
    axes[2].set_ylim([-0.3, 0.3])
    axes[0].set_ylabel(r'$R_{\theta \theta}$')
    axes[0].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$R_{pp}$')
    axes[1].set_xlabel(r'$t$')
    axes[2].set_ylabel(r'$R_{\theta p}$')
    axes[2].set_xlabel(r'$t$')
    plt.tight_layout()
    return theta_corrs, p_corrs, cross_corrs 

def plot_corr_ft(data, dt): 
    fig, axs = plt.subplots(1, 4, sharex=True, figsize=(15, 5))

    omegas = np.fft.rfftfreq(len(data[0][0]))*2*np.pi
    
    for (theta, p) in data:
        theta_omega = np.fft.rfft(theta)
        p_omega = np.fft.rfft(p)
        theta_corr = np.abs(theta_omega)**2
        p_corr = np.abs(p_omega)**2
        cross_corr = np.conj(theta_omega)*p_omega
        
        axs[0].plot(omegas, theta_corr, 'o--', alpha=0.5, color='orange')
        axs[1].plot(omegas, p_corr, 'o--', alpha=0.5, color='orange')
        axs[2].plot(omegas, cross_corr.real, 'o--', alpha=0.5, color='orange')
        axs[3].plot(omegas, cross_corr.imag, 'o--', alpha=0.2, color='orange')
    
    theta_corr = np.mean(np.abs(np.fft.rfft(data[:, 0]))**2, axis=0)
    axs[0].plot(omegas, theta_corr, color='red')
    axs[0].set_title('theta auto corr')

    p_corr = np.mean(np.abs(np.fft.rfft(data[:, 1]))**2, axis=0)
    axs[1].plot(omegas, p_corr, color='red')
    axs[1].set_title('polarisation auto corr')
    
    cross_corr = np.mean(np.conjugate(np.fft.rfft(data[:, 0]))*np.fft.rfft(data[:, 1]), axis=0)
    axs[2].plot(omegas, cross_corr.real, 'x--', color='red')
    
    axs[2].set_title('cross corr real')

    axs[3].plot(omegas, cross_corr.imag, color='red')
    axs[3].set_title('cross corr imag')
    
    axs[0].set_xlim([0, max(omegas[:100])])
    plt.tight_layout()
    plt.show()  

    return theta_corr, p_corr, cross_corr 

def show(data): 
    fig, axes = plt.subplots(1, len(data), sharex=True, sharey=True, figsize=(20, 3))

    for (i, d) in enumerate(data): 
        axes[i].plot(d[1], label='p')
        axes[i].plot(d[0], label='theta')
    axes[0].set_xlim([0, 5000])
    plt.legend()
    plt.show()

def plot_dist(data, color='darkorange'):
   
    fig, axes = plt.subplots(3, 8, sharey='row', sharex='row', figsize=(20, 8))
    for (i, d) in enumerate(data): 
        axes[0, i].hist2d(d[0], d[1], bins=20, color=color)
        axes[0, i].set_xlabel('theta')
        axes[0, i].set_ylabel('p')
        axes[1, i].hist(d[1], bins=20, color=color, density=True)
        axes[2, i].hist(d[0], bins=20, color=color, density=True)
    axes[1, 0].set_ylabel('p pdf')
    axes[2, 0].set_ylabel('theta pdf')
    plt.tight_layout()
    plt.show() 

def plot_overall_dist(data, tex=False, color='darkorange', cmap='Greys'): 
    plt.rc('text', usetex=tex)
    plt.rc('font', family='sans serif', size=20)
    
    thetas = np.concatenate([d[0] for d in data])
    ps = np.concatenate([d[1] for d in data])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _, _, _, im = axes[2].hist2d(thetas, ps, density=True, cmap=cmap, bins=30)
    axes[2].set_xlabel(r'$\theta$')
    axes[2].set_ylabel(r'$p$')
    axes[2].set_xlim([-0.2, 0.2])
    axes[2].set_ylim([-1, 1])

    axes[1].hist(ps, bins=40, density=True, color=color, alpha=0.8)
    axes[1].set_xlim([-1, 1])
    axes[1].set_xlabel(r'$p$')
    axes[1].set_ylabel(r'$P(p)$')
    axes[1].set_ylim([0, 0.9])
    
    axes[0].hist(thetas, bins=40, density=True, color=color, alpha=0.8)
    axes[0].set_xlim([-0.5, 0.5])
    axes[0].set_ylim([0, 9])
    axes[0].set_xlabel(r'$\theta$')
    axes[0].set_ylabel(r'$P(\theta)$')
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label(r'$P(\theta, p)$', rotation=90)
    plt.tight_layout()
    return fig, axes 

def coarsen(array, N): 
    L = int(len(array)/N)
    reshaped_array = np.reshape(array[:L*N], (L, N))
    return np.mean(reshaped_array, axis=-1)
    
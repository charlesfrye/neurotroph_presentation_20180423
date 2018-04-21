import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy

from mpl_toolkits.mplot3d import Axes3D
import time
import os
import string

from scipy.signal import convolve

## LOSS SURFACE VISUALIZATION

def gauss_random_field(x,y,scale):
    white_field = np.random.standard_normal(size=x.shape)
    
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    gauss_rv = scipy.stats.multivariate_normal([0,0],cov=np.ones(2))
    gauss_pdf = gauss_rv.pdf(pos)
    red_field = scale*convolve(white_field,gauss_pdf,mode='same')
    return red_field

def plot_loss_surface(loss,N,mesh_extent):
    mesh = np.linspace(-mesh_extent,mesh_extent,N)
    weights1,weights2 = np.meshgrid(mesh,mesh)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax._axis3don = False

    ax.plot_surface(weights1,weights2,loss(weights1,weights2),
                rstride=2,cstride=2,linewidth=0.05,edgecolor='b',
                alpha=1,cmap='Blues',shade=True);
    
    axis_equal_3d(plt.gca(),center=True)
    
def axis_equal_3d(ax,center=0):
    # FROM StackO/19933125
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    if center == 0:
        centers = [0,0,0]
    else:
        centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

## RANDOM MATRIX GENERATION

def generate_gaussian(N):
    """generate an N by N gaussian random matrix with variance N
    """
    return 1/np.sqrt(N)*np.random.standard_normal(size=(N,N)).astype(np.float32)

def generate_symmetric(N):
    """generate an N by N symmetric gaussian random matrix with variance N
    """
    base_matrix = generate_gaussian(N)
    return (1/np.sqrt(2))*(base_matrix+base_matrix.T)

def generate_wishart(N, k=1):
    """generate an N by N wishart random matrix with rank min(N,k)
    """
    self_outer_product = lambda x: x.dot(x.T)
    wishart_random_matrix = 1/k*self_outer_product(np.random.standard_normal(size=(N,k))).astype(np.float32)

    return wishart_random_matrix

## RANDOM MATRIX VISUALIZATION

def wigner_semicircle(lam):
    return 1/(2*np.pi)*np.sqrt(2**2-lam**2)

def plot_wigner_comparison(eigvals, ax):
    N = len(eigvals)
    sns.distplot(eigvals, kde=False, bins=max(N//20,10),
                 hist_kws={"normed":True, "histtype":"step", "linewidth":8, "alpha":0.8},
                label="empirical spectral density",
                ax=ax);
    
    lams = np.linspace(-2, 2, 100);
    ax.plot(lams, wigner_semicircle(lams),linewidth=8, label="expected spectral density");
    ax.set_ylabel(r"$\rho\left(\lambda\right)$", fontsize=24); ax.set_xlabel(r"$\lambda$", fontsize=24);
    ax.legend(fontsize=16, loc=8);

def marchenkopastur_density(x, N, k, sigma=1):
    """the density for the non-singular portion of the marchenko-pastur distribution,
    as given by https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution.
    """
    lam = N/k
    scaling_factor = 1/(2*np.pi*sigma**2)

    lam_plus = sigma**2*(1+np.sqrt(lam))**2
    lam_minus = sigma**2*(1-np.sqrt(lam))**2

    if (x>lam_minus and x<lam_plus):
        return scaling_factor*(np.sqrt((lam_plus-x)*(x-lam_minus))/(lam*x))
    else:
        return 0

def marchenkopastur_cumulative_distribution(xs, N, k):
    """the cumulative distribution for the marchenko-pastur distribution,
    calculated by numerically integrating the density for the non-singular portion
    and then adding the singular part.
    valid for xs>=0."""
    return max((1-k/N), 0)+scipy.integrate.cumtrapz([marchenkopastur_density(x, N, k) for x in xs],xs)

def plot_marchenko_comparison(eigvals, ax, N, k, eps=1e-6):
    """compare the histogram of the eigvals from a matrix
    to the marchenko-pastur distribution with parameters N, k, and sigma=1.
    the marchenko-pastur distribution is plotted from eps to max(eigvals) with precision 1e-5.
    """
    if k >= N:
        cumulative = False
        legend_label = "empirical spectral density"
        ylabel = r"$\rho\left(\lambda\right)$"
    else:
        cumulative = True
        legend_label = "empirical cumulative spectral distribution"
        ylabel = r"$P(\Lambda \leq \lambda)$"
        
    sns.distplot(eigvals, kde=False, bins=max(len(eigvals)//10,10),
                     hist_kws={"normed":True, "histtype":"step", "linewidth":8, "alpha":0.8,
                              "cumulative":cumulative},
                    label=legend_label,
                ax=ax);
    
    xs = np.linspace(eps, max(eigvals),num=int(1e5))
    if k >=N:
        ax.plot(xs, [marchenkopastur_density(x, N, k) for x in xs],  linewidth=4,
            label="expected spectral density")
        ax.legend(fontsize=16, loc="upper right");
    else:
        ax.plot(xs[:-1], marchenkopastur_cumulative_distribution(xs, N, k), linewidth=4,
            label="expected cumulative spectral distribution");
        ax.legend(fontsize=16, loc="lower right");

    
    plt.tick_params(labelsize=16)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xlabel(r"$\lambda$", fontsize=24)
    
    
def plot_matrix(matrix, ax):
    ax.imshow(matrix,cmap="Greys"); plt.axis('off');
    
def plot_matrix_and_spectrum(matrix, spectrum_plotter, spectrum_plotter_kwargs={}):
    f = plt.figure(figsize=(12,4)); axs_shape=(1,3)
    mat_ax = plt.subplot2grid(shape=axs_shape, loc=(0,0));
    plot_matrix(matrix, mat_ax)
    eigvals = np.linalg.eigvals(matrix)
    
    spec_ax = plt.subplot2grid(shape=axs_shape, loc=(0,1), colspan=2)
    spectrum_plotter(eigvals, spec_ax, **spectrum_plotter_kwargs)
    plt.tight_layout()

## CURVATURE VISUALIZATION

def plot_curvature_training(indices, cost, spectrums, k=3):

    f, axs = plt.subplots(2,k, figsize=(6*k,6))
    [ax.set_yscale('log') for ax in axs.flatten()]
    
    for col_idx, ax_col in enumerate(axs.T):
        cost_ax, spec_ax = ax_col
        index = col_idx*(len(indices)-1)//(k-1)
        
        cost_ax.semilogy(indices, cost, linewidth=4);
        cost_ax.set_ylabel("cost", fontsize=18); cost_ax.set_xlabel("training time", fontsize=18);
        
        tracker = cost_ax.scatter(indices[index], cost[index], color='r', s=36, zorder=12)

        cost_ax.set_xticklabels([int(tick) for tick in cost_ax.get_xticks()], fontdict={"fontsize":"x-large"})
        cost_ax.set_yticklabels(cost_ax.get_yticks(), fontdict={"fontsize":"x-large"})
        
        mn, mx = -0.5, 1.5
        bins = np.linspace(-0.5, 1.5)
        _, _, patches = spec_ax.hist(spectrums[index], histtype="step", linewidth=4, bins=bins, color='k');
        spec_ax.autoscale(enable=False)
        xlims = spec_ax.get_xlim(); ylims = spec_ax.get_ylim();
        
        spec_ax.set_ylabel("count", fontsize=18); spec_ax.set_xlabel(r"$\lambda$", fontsize=18);
        spec_ax.set_xticks([-0.5,0,0.5,1]); spec_ax.set_xticklabels(spec_ax.get_xticks(), fontdict={"fontsize":"x-large"})
        spec_ax.set_yticklabels([int(tick) for tick in spec_ax.get_yticks()], fontdict={"fontsize":"x-large"})
        plt.tight_layout()
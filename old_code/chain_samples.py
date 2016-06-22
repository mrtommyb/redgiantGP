#!/usr/bin/env python
# -*- coding: utf-8 -*-
## turn koi2133 chains into data model

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import george
from george.kernels import ExpSquaredKernel, RBFKernel, ExpKernel

from ktransit import LCModel, FitTransit

from scipy import optimize

import h5py

def get_sample(chain_vals,time,flux,ferr,rvtime):
    M = LCModel()
    M.add_star(rho=chain_vals[0],zpt=chain_vals[1],ld1=chain_vals[2],
        ld2=chain_vals[3],veloffset=chain_vals[4])
    M.add_planet(T0=chain_vals[7],period=chain_vals[8],impact=chain_vals[9],
        rprs=chain_vals[10],ecosw=chain_vals[11],esinw=chain_vals[12],
        rvamp=chain_vals[13],occ=chain_vals[14],ell=chain_vals[15],alb=chain_vals[16])
    M.add_data(time=time)
    M.add_rv(rvtime=rvtime)

    #kernel = ((chain_vals[5]**2 * RBFKernel(chain_vals[6])) +
    #        (chain_vals[7]**2 * RBFKernel(chain_vals[8])))
    #kernel = ((chain_vals[5]**2 * ExpKernel(chain_vals[6])) +
    #        (chain_vals[7]**2 * RBFKernel(chain_vals[8])))
    kernel = chain_vals[5]**2 * RBFKernel(chain_vals[6])
    gp = george.GaussianProcess(kernel)

    sample = np.array([])
    for i in np.arange(len(time) // 1000):
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], ferr[:][section])
        sample = np.r_[sample,gp.predict(
            flux[:][section] - M.transitmodel[section],time[section])[0]]
    return sample, M.transitmodel

def get_many_samples(chain_vals,time,flux,ferr,rvtime,nsamples=300):
    M = LCModel()
    M.add_star(rho=chain_vals[0],zpt=chain_vals[1],ld1=chain_vals[2],
        ld2=chain_vals[3],veloffset=chain_vals[4])
    M.add_planet(T0=chain_vals[7],period=chain_vals[8],impact=chain_vals[9],
        rprs=chain_vals[10],ecosw=chain_vals[11],esinw=chain_vals[12],
        rvamp=chain_vals[13],occ=chain_vals[14],ell=chain_vals[15],alb=chain_vals[16])
    M.add_data(time=time)
    M.add_rv(rvtime=rvtime)

    #kernel = ((chain_vals[5]**2 * RBFKernel(chain_vals[6])) +
    #        (chain_vals[7]**2 * RBFKernel(chain_vals[8])))
    #kernel = ((chain_vals[5]**2 * ExpKernel(chain_vals[6])) +
    #        (chain_vals[7]**2 * RBFKernel(chain_vals[8])))
    kernel = chain_vals[5]**2 * RBFKernel(chain_vals[6])
    gp = george.GaussianProcess(kernel)

    slist = np.arange(len(time) // 1000)
    samples = np.zeros([nsamples,len(slist)*1000])
    for i in slist:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], ferr[:][section])
        samples[:,section] = gp.sample_conditional(
            flux[:][section] - M.transitmodel[section],time[section],
            size=nsamples)
    return samples, M.transitmodel


def get_rv():
    pass

if __name__ == '__main__':
    fn = 'koi2133_np1_priorTrue_dil0.0GP.hdf5'
    f = h5py.File(fn)
    g = f['mcmc']['chain'][:]
    lnprob = f['mcmc']['lnprob'][:]

    mle_idx = np.unravel_index(lnprob.argmax(),
            lnprob.shape)

    mle = g[mle_idx]

    time = f['time'][:]
    flux = f['flux'][:]
    ferr = f['err'][:]
    rvtime = f['rvtime']

    doplot = True
    plot_many = False
    if plot_many and doplot:
        samples, tmod = get_many_samples(mle,time,flux,ferr,rvtime,
            nsamples=300)

        time = f['time'][:61000]
        flux = f['flux'][:61000]
        tmod1 = tmod[:61000]

        fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=False,
            figsize=[9,6])
        ax1.scatter(time,flux,s=3,label='Kepler data')
        med = np.median(samples+tmod1,axis=0)
        stdm = np.std(samples+tmod1,axis=0)
        ax1.plot(time,med,color='purple',
                label='Light curve + noise model',lw=1,
                alpha=0.6)
        ax1.fill_between(time,med-stdm,med+stdm,
            alpha=0.3,color='purple')
        ax1.set_xlim([583.8,596.9])
        ax1.legend()
        ax1.set_xlabel('Time (BJD-2454833)',labelpad=12)
        ax1.set_ylabel('Relative flux')
        ax1.minorticks_on()


    elif not plot_many and doplot:
        sample, tmod = get_sample(mle,time,flux,ferr,rvtime)

        time = f['time'][:61000]
        flux = f['flux'][:61000]
        tmod1 = tmod[:61000]
        fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=False,
            figsize=[9,6])
        ax1.scatter(time,flux,s=1,label='Kepler data')
        ax1.plot(time,sample,color='b',label='Noise model')
        ax1.plot(time,tmod1,color='r',
            label='Light curve model')
        ax1.plot(time,sample+tmod1,color='purple',
            label='Light curve + noise model',lw=2)
        ax1.set_xlim([583.8,596.9])
        ax1.legend()
        ax1.set_xlabel('Time (BJD-2454833)',labelpad=12)
        ax1.set_ylabel('Relative flux')
        ax1.minorticks_on()

    many_plots = True
    if many_plots:
        period = mle[8]
        T0 = mle[7]
        fig, (axes) = plt.subplots(5, 2, sharex=False, sharey=True,
            figsize=[7,14])
        axes = axes.flatten()
        for i,offset in enumerate(T0 + period*np.arange(67,77,1)):
            if i == 2:
                offset = T0 + period*77
            elif i == 7:
                offset = T0 + period*78
            win = 1.3
            trange = (time < offset+win) & (time > offset-win)

            axes[i].scatter(time[trange],flux[trange]
                ,s=1,label='Kepler data')
            axes[i].plot(time[trange],sample[trange],
                color='b',label='Noise model')
            axes[i].plot(time[trange],tmod1[trange],
                color='r',lw=1.5,
                label='Light curve model')
            axes[i].plot(time[trange],sample[trange]+tmod1[trange],
                color='purple',
                label='Light curve + noise model',lw=2)
            axes[i].set_xlim([offset-win,offset+win])
            axes[i].tick_params(labelbottom=False,labelleft=False)
            axes[i].minorticks_on()
        axes[i].set_ylim([-0.0017,0.0012])
        plt.subplots_adjust(wspace=0, hspace=0)











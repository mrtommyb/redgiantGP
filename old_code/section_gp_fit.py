#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import george
from george.kernels import ExpSquaredKernel, RBFKernel

from ktransit import LCModel, FitTransit

from scipy import optimize

def ret_opt(params,time,flux,yerr):
    period, T0, rprs, impact, noiseA, noiseW = params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = noiseA * ExpSquaredKernel(noiseW)
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 1000)[0:10]:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_product(params,time,flux,yerr):
    (period, T0, rprs, impact, noiseA1, noiseW1,
        noiseA2, noiseW2, noiseM2) = params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = (((noiseA1 * ExpSquaredKernel(noiseW1)) *
            (noiseA2 * ExpSquaredKernel(noiseW2))) + noiseM2)
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 1000)[0:10]:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_sum(params,time,flux,yerr):
    (period, T0, rprs, impact,
        alb,occ,ell,ecosw, esinw,
        noiseA1, noiseW1,
        noiseA2, noiseW2 )= params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = ((noiseA1**2 * RBFKernel(noiseW1)) +
            (noiseA2**2 * RBFKernel(noiseW2)))
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 1000)[0:10]:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_sum_ln(params,time,flux,yerr):
    (period, T0, rprs, impact,
        alb,occ,ell,ecosw, esinw,
        lnnoiseA1, lnnoiseW1,
        lnnoiseA2, lnnoiseW2 )= params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = ((np.exp(lnnoiseA1)**2 * RBFKernel(np.exp(lnnoiseW1))) +
            (np.exp(lnnoiseA2)**2 * RBFKernel(np.exp(lnnoiseW2))))
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 1000)[0:10]:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_simple_ln(params,time,flux,yerr):
    (period, T0, rprs, impact,
        alb,occ,ell,ecosw, esinw,
        lnnoiseA1, lnnoiseW1)= params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = (np.exp(lnnoiseA1)**2 * RBFKernel(np.exp(lnnoiseW1)))
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 1000):
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_simple(params,time,flux,yerr):
    (period, T0, rprs, impact,
        alb,occ,ell,ecosw, esinw,
        noiseA1, noiseW1)= params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = (noiseA1**2 * RBFKernel(noiseW1))
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 300):
        section = np.arange(i*300,i*300 + 300)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

def ret_simplest(params,time,flux,yerr,fixed):
    (period, T0, rprs, impact,
            alb,occ,ell,ecosw, esinw) = fixed
    (noiseA1, noiseW1)= params
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)
    resid = flux - M.transitmodel
    kernel = (noiseA1**2 * RBFKernel(noiseW1))
    gp = george.GaussianProcess(kernel)
    lnlike = 0.
    for i in np.arange(len(time) // 300):
        section = np.arange(i*300,i*300 + 300)
        gp.compute(time[section], yerr[section])
        lnlike += gp.lnlikelihood(resid[section])
    return -lnlike

if __name__ == '__main__':
    data = np.genfromtxt('/Users/tom/Projects/koi2133/data/lc.dat').T
    time = data[0] #test on shorter data set
    flux = data[1]
    ferr = (data[2] / 4.)

    product = False
    sumkernel = False
    simple = False
    even_simpler = True
    if not product and not sumkernel and not simple and not even_simpler:
        ## vary the period, T0, rprs, b, noiseA, noiseW
        bounds = ((None,None),(None,None),
            (0.01,0.04),(0.00001,0.999),(None,None),(None,None))
        guess = (6.24658,136.3966,0.02255,0.5,0.05,0.01)
        lsqout = optimize.fmin_l_bfgs_b(ret_opt,guess,
            args=(time,flux,ferr),approx_grad=True,bounds=bounds)

        period, T0, rprs, impact, noiseA, noiseW = lsqout[0]
        kernel = noiseA * ExpSquaredKernel(noiseW)
        gp = george.GaussianProcess(kernel)

    elif product:
        ## vary the period, T0, rprs, b, noiseA1, noiseW1,
        ## noiseA2, noiseW2, noiseM2
        bounds = ((None,None),(None,None),
            (0.01,0.04),(0.00001,0.999),(None,None),(None,None),
            (None,None),(None,None),(None,None))
        guess = (6.24658,136.3966,0.02255,0.5,0.01,6.,
            0.01,0.12,0.0)
        lsqout = optimize.fmin_l_bfgs_b(ret_product,guess,
            args=(time,flux,ferr),approx_grad=True,bounds=bounds,m=100,factr=1.E6)

        (period, T0, rprs, impact, noiseA1, noiseW1,
            noiseA2, noiseW2, noiseM2) = lsqout[0]
        kernel = (((noiseA1 * ExpSquaredKernel(noiseW1)) *
            (noiseA2 * ExpSquaredKernel(noiseW2))) + noiseM2)
        gp = george.GaussianProcess(kernel)

    elif sumkernel:
        ## vary the period, T0, rprs, b,
        ## alb,occ,ell,ecosw, esinw
        ## noiseA1, noiseW1,
        ## noiseA2, noiseW2
        bounds = ((None,None),(None,None),
            (0.01,0.03),(0.00001,0.999),
            (None,None),(None,None),
            (None,None),(None,None),(None,None),
            (None,None),(None,None),
            (None,None),(None,None))
        guess = (6.24658,136.3966,0.02255,0.8,
            30., 30., 60., 0.0, 0.0,
            np.log(5.E-4), np.log(0.07),
            np.log(2.E-4), np.log(3.0))
        lsqout = optimize.fmin_l_bfgs_b(ret_sum_ln,guess,
            args=(time,flux,ferr),approx_grad=True,bounds=bounds,m=300,factr=1.E7)

        (period, T0, rprs, impact,
            alb,occ,ell,ecosw, esinw,
            lnnoiseA1, lnnoiseW1,
            lnnoiseA2, lnnoiseW2) = lsqout[0]
        kernel = ((np.exp(lnnoiseA1)**2 * RBFKernel(np.exp(lnnoiseW1))) +
            (np.exp(lnnoiseA2)**2 * RBFKernel(np.exp(lnnoiseW2))))
        gp = george.GaussianProcess(kernel)

    elif simple:
        bounds = ((None,None),(None,None),
            (0.01,0.03),(0.00001,0.999),
            (None,None),(None,None),
            (None,None),(None,None),(None,None),
            (None,None),(None,None))
        guess = (6.2465796,136.39661,0.02255,0.4,
            30., 30., 60., 0.0, 0.0,
            5.E-4, 0.07)
        lsqout = optimize.fmin_l_bfgs_b(ret_simple,guess,
            args=(time,flux,ferr),approx_grad=True,bounds=bounds,
            m=300,factr=1.E7)

        (period, T0, rprs, impact,
            alb,occ,ell,ecosw, esinw,
            noiseA1, noiseW1) = lsqout[0]
        kernel = (noiseA1**2 * RBFKernel(noiseW1))
        gp = george.GaussianProcess(kernel)

    elif even_simpler:
        bounds = (
            (None,None),(None,None))
        guess = (5.E-4, 0.07)
        fixed = (6.2465796,136.39661,0.02255,0.8,
            13., 30., 45., 0.0, 0.0)
        (period, T0, rprs, impact,
            alb,occ,ell,ecosw, esinw) = fixed
        lsqout = optimize.fmin_l_bfgs_b(ret_simplest,guess,
            args=(time,flux,ferr,fixed),approx_grad=True,bounds=bounds,
            m=300,factr=1.E7)

        (noiseA1, noiseW1) = lsqout[0]
        kernel = (noiseA1**2 * RBFKernel(noiseW1))
        gp = george.GaussianProcess(kernel)

    #transit fit
    M = LCModel()
    M.add_star(rho=0.0073,ld1=0.5,ld2=0.4)
    M.add_planet(T0=T0,period=period,
        impact=impact,rprs=rprs,
        alb=alb,occ=occ,ell=ell,
        ecosw=ecosw, esinw=esinw)
    M.add_data(time=time)

    #sample = np.array([])
    #for i in np.arange(len(time) // 1000):
    #    section = np.arange(i*1000,i*1000 + 1000)
    #    gp.compute(time[section], ferr[section])
    #    sample = np.r_[sample,gp.sample_conditional(
    #        flux[section] - M.transitmodel[section],time[section])]

    #gp.compute(time, ferr)
    #samples = gp.sample_conditional(flux, time, N=100)

    print('here')

    sample = np.array([])
    for i in np.arange(len(time) // 300):
        section = np.arange(i*300,i*300 + 300)
        gp.compute(time[section], ferr[section])
        sample = np.r_[sample,gp.predict(
            flux[section] - M.transitmodel[section],time[section])[0]]






#import transitemcee
from transitemcee import get_ar
import numpy as np
from copy import deepcopy
from scipy.stats import truncnorm
from claretquadpy import claretquad
from claret4ppy import claretlimb4p
from numpy import random
import time as thetime
import emcee
import tmodtom as tmod
import sys

import transitemcee_rv

from scipy import optimize, fftpack, signal

import george
from george.kernels import ExpSquaredKernel

class transitemcee_koi2133(transitemcee_rv.transitemcee_rv):

    def __init__(self,nplanets,cadence=1625.3,rvcadence=1800.0,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        super(transitemcee_koi2133,self).__init__(nplanets,cadence=cadence,
            ldfileloc=ldfileloc,
            codedir=codedir)
        sys.path.append(codedir)
        self.rvcadence = rvcadence / 86400.


    def get_sol(self,*args,**kwargs):
        """
        like get_sol from regular transitemcee
        but with an extra variable for every
        planet -- rvamp
        also veloffset is impoartant now

        even more new things for this one
        now need alb, occ ell

        even morer parameters
        4 parameters for the noise model
        2 for each kernel
        """
        tom = args
        assert np.shape(tom)[0] == self.nplanets * (7 + 3)

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print ' running with dil = %s' %(dil)
        else:
            dil = 0.0

        if 'veloffset' in kwargs.keys():
            self.veloffset = kwargs['veloffset']
        else:
            self.veloffset = 0.0

        #occ = 0.0
        #alb = 0.0
        #ell = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6

        if 'noise_model' in kwargs.keys():
            self.noise_model = kwargs['noise_model']
        else:
            self.noise_model = self.get_noise_model()

        (self.Lamp,self.Lwid,self.Gamp,self.Gwid) = self.noise_model

        fit_sol = np.array([self.rho_0,self.zpt_0,
            self.ld1,self.ld2,self.veloffset,
            self.Lamp,self.Lwid,self.Gamp,self.Gwid])

        for i in xrange(self.nplanets):
            T0_0 = args[i*7]
            per_0 = args[i*7 +1]
            b_0 = args[i*7 +2]
            rprs_0 = args[i*7 +3]
            ecosw_0 = args[i*7 +4]
            esinw_0 = args[i*7 +5]
            rvamp_0 = args[i*7 + 6]
            occ_0 = args[i*7 + 7]
            ell_0 = args[i*7 + 8]
            alb_0 = args[i*7 + 9]

            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0,rvamp_0,
                occ_0,ell_0,alb_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)

        self.fixed_sol = np.array([
            dil])

    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw,rvamp,occ,ell,alb
        """

        rho_unc = 0.001
        zpt_unc = 1.E-9
        ld1_unc = 0.001
        ld2_unc = 0.001
        veloffset_unc = 2.
        T0_unc = 0.0002
        per_unc = 0.00001
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        rvamp_unc = 0.01
        occ_unc = 0.01
        ell_unc = 0.01
        alb_unc = 0.01
        Lamp_unc = 0.01
        Lwid_unc = 0.01
        Gamp_unc = 0.01
        Gwid_unc = 0.01

        if self.n_ldparams == 2:
            p0 = np.zeros([nwalkers,9+self.nplanets*10+2])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]
        veloffset = self.fit_sol[4]
        Lamp = self.fit_sol[5]
        Lwid = self.fit_sol[6]
        Gamp = self.fit_sol[7]
        Gwid = self.fit_sol[8]

        start,stop = ((0.000001 - rho) / rho_unc,
            (30.0 - rho) / rho_unc)
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,
            size=nwalkers)

        start,stop = ((0.0 - ld1) / ld1_unc,
            (1.0 - ld1) / ld1_unc)
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = ((0.0 - ld2) / ld2_unc,
            (1.0 - ld2) / ld2_unc)
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)

        start,stop = ((-30. - veloffset) / veloffset_unc,
            (30. - veloffset) / veloffset_unc)
        p0[...,4] = truncnorm.rvs(start,stop
                ,loc=veloffset,scale=veloffset_unc,size=nwalkers)

        #noise parameters
        start,stop = ((-8.9 - Lamp) / Lamp_unc,
            (-8.1 - Lamp) / Lamp_unc)
        p0[...,5] = truncnorm.rvs(start,stop
                ,loc=Lamp,scale=Lamp_unc,size=nwalkers)

        start,stop = ((-6.9 - Lwid) / Lwid_unc,
            (-6.1 - Lwid) / Lwid_unc)
        p0[...,6] = truncnorm.rvs(start,stop
                ,loc=Lwid,scale=Lwid_unc,size=nwalkers)

        start,stop = ((-20. - Gamp) / Gamp_unc,
            (-1. - Gamp) / Gamp_unc)
        p0[...,7] = truncnorm.rvs(start,stop
                ,loc=Gamp,scale=Gamp_unc,size=nwalkers)

        start,stop = ((-20. - Gwid) / Gwid_unc,
            (-1. - Gwid) / Gwid_unc)
        p0[...,8] = truncnorm.rvs(start,stop
                ,loc=Gwid,scale=Gwid_unc,size=nwalkers)

        for i in xrange(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw,rvamp,occ,ell,alb) = self.fit_sol[i*7+5+4:i*7+15+4]
            b = 0.9
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*7+5+4] = np.random.normal(
                T0,T0_unc,size=nwalkers)

            p0[...,i*7+6+4] = np.random.normal(
                per,per_unc,size=nwalkers)
            
            start,stop = (0.0 - b) / b_unc, (1.0 - b) / b_unc
            p0[...,i*7+7+4] = truncnorm.rvs(
                start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*7+8+4] = truncnorm.rvs(
                start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            
            start,stop = (-0.2 - ecosw) / ecosw_unc, (0.2 - ecosw) / ecosw_unc
            p0[...,i*7+9+4] = truncnorm.rvs(
                start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            
            start,stop = (-0.2 - esinw) / esinw_unc, (0.2 - esinw) / esinw_unc
            p0[...,i*7+10+4] = truncnorm.rvs(
                start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)
            
            start,stop = (3 - rvamp) / rvamp_unc, (7 - rvamp) / rvamp_unc
            p0[...,i*7+11+4] = truncnorm.rvs(
                start,stop
                ,loc=rvamp,scale=rvamp_unc,size=nwalkers)


            start,stop = (-2 - occ) / occ_unc, (4 - occ) / occ_unc
            p0[...,i*7+12+4] = truncnorm.rvs(
                start,stop
                ,loc=occ,scale=occ_unc,size=nwalkers)

            start,stop = (-2 - ell) / ell_unc, (4 - ell) / ell_unc
            p0[...,i*7+13+4] = truncnorm.rvs(
                start,stop
                ,loc=ell,scale=ell_unc,size=nwalkers)

            start,stop = (-2 - alb) / alb_unc, (4. - alb) / alb_unc
            p0[...,i*7+14+4] = truncnorm.rvs(
                start,stop
                ,loc=alb,scale=alb_unc,size=nwalkers)

        #lcjitter
        start,stop = 0.0, 10.
        p0[...,-2] = truncnorm.rvs(start,stop,
            loc=-4.,scale=0.1,size=nwalkers)

        #rvjitter
        start,stop = 0.0, 10.
        p0[...,-1] = truncnorm.rvs(start,stop,
            loc=0,scale=0.1,size=nwalkers)
        return p0

    def get_noise_model(self):
        """
        this is a good noise model guess for epic 2113.
        """
        #noise_model = np.array([1.13150997e-01, 1.43286469e+00,
        #    1.08756091e-02,
        #    1.37080589e+00, 9.63791885e+00])
        noise_model = [
            np.log(1.6E-4), np.log(4.4E-2**2), 2.E-4, 3.]
        return noise_model

def logchi2_rv_phaseGP2(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    rvtime,rvval,rverr,rvitime,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0,
    use_hodlr=False):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2,veloffset
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0,rvamp_0,occ_0,ell_0,alb_0

    fixed_sol should have
    dil

    """
    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    zpt = fitsol[1]
    if np.abs(zpt) > 1.E-2:
        return minf


    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0

    # time to anti-log things
    expGP1 = fitsol[5]
    expGP2 = fitsol[6]
    GP1 = np.exp(fitsol[5])
    GP2 = np.exp(fitsol[6])
    rvamp = np.exp(fitsol[np.arange(nplanets)*7 + 13])
    occ = np.exp(fitsol[np.arange(nplanets)*7 + 14])
    ell = np.exp(fitsol[np.arange(nplanets)*7 + 15])
    alb = np.exp(fitsol[np.arange(nplanets)*7 + 16])
    jitter_lc = np.exp(fitsol[-2])
    jitter_rv = np.exp(fitsol[-1])

    if GP1 < 0.0 or GP1 > 10.0:
    #    print('should never execute, GP1<0.0')
        return minf
    if GP2 < 0.0 or GP2 > 10.0:
    #    print('should never execute, GP2<0.0')
        return minf

    if expGP1 < -11. or expGP1 > -6.:
    #    print('expGP1 == {}'.format(expGP1))
        return minf
    if expGP2 < -8. or expGP2 > -4.:
     #   print('expGP2 == {}'.format(expGP2))
        return minf


    #T0, period, b, rprs, ecosw, esinw

    rprs = fitsol[np.arange(nplanets)*7 + 10]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*7 + 11]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*7 + 12]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*7 + 8]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*7 + 9]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*7 + 7]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf

    if np.abs(jitter_rv > 500):
        return minf

    veloffset = fitsol[4]
    if np.abs(veloffset) > 200:
        return minf

    if np.abs(rvamp) > 1.E6 or rvamp < 0.0:
    #    print('should rarely execute, rvamp  == {}'.format(rvamp))
        return minf
    if np.abs(occ) > 1.E6 or occ < 0.:
    #    print('should rarely execute, occ  == {}'.format(occ))
        return minf
    if np.abs(ell) > 1.E6 or ell < 0.:
    #    print('should rarely execute, ell  == {}'.format(ell))
        return minf
    if np.abs(alb) > 1.E6 or alb < 0.:
    #    print('should rarely execute, alb  == {}'.format(alb))
        return minf
    if ecc > 0.6:
        return minf


    if jitter_lc < 0.0 or jitter_lc > 0.7:
    #    print('should rarely execute, lcjutter  == {}'.format(jitter_lc))
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2


    if jitter_rv < 0.0 or jitter_rv > 500.:
    #    print('should never execute, rv jitter<0.0')
        return minf
    rverr_jit = np.sqrt(rverr**2 + jitter_rv**2)
    rverr_jit2 = rverr**2 + jitter_rv**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4],fitsol[7:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = np.r_[time,rvtime]
    itime_model_calc = np.r_[itime,rvitime]
    datatype_model_calc  = np.r_[datatype,np.ones_like(rvtime)]

    model_lcrv = calc_model_phase(fitsol_model_calc,
        nplanets,fixed_sol_model_calc,
        time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv[datatype_model_calc == 0] - 1.
    model_rv = model_lcrv[datatype_model_calc == 1]


    ecc[ecc == 0.0] = 1.E-10

    npt_lc = len(err_jit)
    npt_rv = len(rverr_jit)

    #do the GP stuff
    resid = flux - model_lc
    kernel = (GP1**2 * ExpSquaredKernel(GP2))
    if use_hodlr:
        gp = george.GP(kernel, solver=george.HODLRSolver)
    else:
        gp = george.GP(kernel)

    lnlike = 0.

    # for i in np.arange(len(time) // 1000):
    #     section = np.arange(i*1000,i*1000 + 1000)
    #     gp.compute(time[section], err_jit[section])
    #     lnlike += gp.lnlikelihood(resid[section])
    gp.compute(time, err_jit)
    lnlike += gp.lnlikelihood(resid)
#    loglc = (
#        - (npt_lc/2.)*np.log(2.*np.pi)
#        - 0.5 * np.sum(np.log(err_jit2))
#        - 0.5 * np.sum((res)**2 / err_jit2)
#        )
    loglc = lnlike

    logrv = (
        - (npt_rv/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(rverr_jit2))
        - 0.5 * np.sum((model_rv - rvval)**2 / rverr_jit2)
        )

    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(rho_0_unc)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        rho_prior = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld1_0_unc)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld2_0_unc)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    logLtot = loglc + logrv + logrho + logldp + logecc

    return logLtot

def calc_model_phase(fitsol,nplanets,fixed_sol,time,itime,ntt,tobs,omc,datatype):
    """
    what is in fitsol??
    rho, zpt, veloffset,

    what is in fixed_sol??
    ld1,ld2,ld3,ld4, dil
    """
    sol = np.zeros([8 + 11*nplanets])
    rho = fitsol[0]
    zpt = fitsol[1]
    ld1,ld2,ld3,ld4 = fixed_sol[0:4]
    dil = fixed_sol[4]
    veloffset = fitsol[2]


    sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt])
    for i in xrange(nplanets):
        sol[8+(i*10):8+(i*10)+10] = fitsol[3+i*7:13+i*7]

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype)

    return tmodout
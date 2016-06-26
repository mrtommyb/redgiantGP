from __future__ import division, print_function

#from clean_and_search import Clean
import numpy as np
from scipy.stats import scoreatpercentile as scpc
import h5py
from scipy.stats import nanmedian, nanstd
#import pyfits
#import kplr

import transitemcee_EPIC as tmod
import emcee
import time as thetime
#import os
from emcee.utils import MPIPool
import sys
from astropy.utils.console import ProgressBar

def get_rv():
    rvdat = np.array([
        [2457414.927751,   14.84,    0.68],
        [2457422.855362,   -17.18,     0.72],
        [2457439.964043,     1.92,     0.82],
        [2457495.743272,   -1.539,   24.133],
        [2457498.729824,  -30.008,   27.269],
        [2457505.670536,  -83.664,   39.032],
        [2457507.723056,   27.440,   29.891],
        [2457520.740837,  -12.297,   31.743],
        [2457524.687701,   -0.044,   31.637],
        [2457525.68652,   67.387,   30.055]
        ])
    rvtime = rvdat[...,0] - 2454833.
    rvval = rvdat[...,1] - np.median(rvdat[...,1])
    rverr = rvdat[...,2]
    return rvtime,rvval,rverr

def get_lc():
    time,flux,ferr = np.genfromtxt(
        '../data/2133_lc_tom.txt',unpack=True)
    flux -= 1.0 #make the median zero
    return time,flux,ferr


def main(runmpi=True,nw=100,th=6,bi=10,fr=10,
    use_hodlr=False):

    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None

    ldfileloc = '/Users/tom/gitcode/redgiantGP/code/'
    codedir = '/Users/tom/gitcode/redgiantGP/code'
    # ldfileloc = '/nobackupp8/tsbarcl2/redgiantGP/code/'        
    # codedir = '/nobackupp8/tsbarcl2/redgiantGP/code'

    koi = 2113
    cadence=1625.3
    teff = 4790.
    teff_unc = 90.
    feh = 0.38
    feh_unc = 0.08
    logg = 3.26
    logg_unc = 0.01
    rho = 0.0222
    rho_unc = 0.0004
    nplanets = 1

    dil=0.0

    period=8.40477896
    impact=9.20759558e-01
    T0=2.30906152e+03
    rprs=3.15254574e-02
    alb=10
    occ=10
    ell=10
    rvamp=103.0
    ecosw=0.048
    esinw=-0.045
    planet_guess = np.array([
        T0,period,impact,rprs,ecosw,esinw,
        rvamp,occ,ell,alb])


    rvtime, rvval, rverr = get_rv()
    time,flux,ferr = get_lc()

    rho_prior = True
    ldp_prior = False

    nwalkers = nw
    threads = th
    burnin = bi
    fullrun = fr
    thin = 1

    n_ldparams = 2

    toffset_lc = 0
    toffset_rv = 0

    zpt_0 = 1.E-10

    M = tmod.transitemcee_koi2133(
        nplanets,cadence,
        ldfileloc=ldfileloc,codedir=codedir)

    M.get_stellar(teff,
        logg,
        feh,
        n_ldparams,ldp_prior=ldp_prior)

    M.already_open(time,
        flux,ferr,
        rvtime,rvval,
        rverr,
        timeoffset=toffset_lc,rvtimeoffset=toffset_rv,
        normalize=False)

    rho_vals = np.array([rho,rho_unc])

    M.get_rho(rho_vals,rho_prior)
    M.get_zpt(zpt_0)

    noise_model = [1.6E-4, 4.4E-2, 2.E-4, 3.]


    if dil is not None:
        M.get_sol(*planet_guess,dil=dil,noise_model=noise_model)
    else:
        M.get_sol(*planet_guess,noise_model=noise_model)


    outfile = 'koi{0}_np{1}_prior{2}_dil{3}GP.hdf5'.format(
            koi,nplanets,rho_prior,dil)

    p0 = M.get_guess(nwalkers)

    #dirty hack!!
    qwe = np.r_[np.arange(0,7),np.arange(9,21)]
    p0 = p0[:,qwe]

    l_var = np.shape(p0)[1]

    N = len([indval for indval in xrange(fullrun)
            if indval%thin == 0])
    with h5py.File(outfile, u"w") as f:
        f.create_dataset("time", data=M.time)
        f.create_dataset("1.6E-4, 4.4E-2, 2.E-4, 3.flux", data=M.flux)
        f.create_dataset("err", data=M.err)
        f.create_dataset("rvtime", data=M.rvtime)
        f.create_dataset("rvval", data=M.rvval)
        f.create_dataset("rverr", data=M.rverr)
        f.create_dataset("itime", data=M._itime)
        f.create_dataset("ntt", data = M._ntt)
        f.create_dataset("tobs", data = M._tobs)
        f.create_dataset("omc",data = M._omc)
        f.create_dataset("datatype",data = M._datatype)
        f.attrs["rho_0"] = M.rho_0
        f.attrs["rho_0_unc"] = M.rho_0_unc
        f.attrs["nplanets"] = M.nplanets
        f.attrs["ld1"] = M.ld1
        f.attrs["ld2"] = M.ld2
        f.attrs["koi"] = koi
        f.attrs["dil"] = dil
        g = f.create_group("mcmc")
        g.attrs["nwalkers"] = nwalkers
        g.attrs["burnin"] = burnin
        g.attrs["iterations"] = fullrun
        g.attrs["thin"] = thin
        g.attrs["rho_prior"] = rho_prior
        g.attrs["ldp_prior"] = ldp_prior
        g.attrs["onlytransits"] = M.onlytransits
        g.attrs["tregion"] = M.tregion
        g.attrs["ldfileloc"] = M.ldfileloc
        g.attrs["n_ldparams"] = M.n_ldparams
        g.create_dataset("fixed_sol", data= M.fixed_sol)
        g.create_dataset("fit_sol_0", data= M.fit_sol_0)


        c_ds = g.create_dataset("chain",
            (nwalkers, N, l_var),
            dtype=np.float64)
        lp_ds = g.create_dataset("lnprob",
            (nwalkers, N),
            dtype=np.float64)

        #I don't like the default LDP unc
        #I'm changing them
        M.ld1_unc = 0.8
        M.ld2_unc = 0.8

        args = [M.nplanets,M.rho_0,M.rho_0_unc,M.rho_prior,
            M.ld1,M.ld1_unc,M.ld2,M.ld2_unc,M.ldp_prior,
            M.flux,M.err,M.fixed_sol,M.time,M._itime,M._ntt,
            M._tobs,M._omc,M._datatype,
            M.rvtime,M.rvval,M.rverr,M._rvitime,
            M.n_ldparams,M.ldfileloc,
            M.onlytransits,M.tregion,use_hodlr]


        tom = tmod.logchi2_rv_phaseGP2


        if runmpi:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,threads=th)


        time1 = thetime.time()
        p2, prob, state = sampler.run_mcmc(p0, burnin,
            storechain=False)
        sampler.reset()
        with h5py.File(outfile, u"a") as f:
            g = f["mcmc"]
            g.create_dataset("burnin_pos", data=p2)
            g.create_dataset("burnin_prob", data=prob)

        time2 = thetime.time()
        print('burn-in took ' + str((time2 - time1)/60.) + ' min')
        time1 = thetime.time()
        pbar = ProgressBar(fullrun)
        for i, (pos, lnprob, state) in enumerate(sampler.sample(p2,
            iterations=fullrun, rstate0=state,
            storechain=False)):
            pbar.update()
            #do the thinning in the loop here
            if i % thin == 0:
                ind = i / thin
                with h5py.File(outfile, u"a") as f:
                    g = f["mcmc"]
                    c_ds = g["chain"]
                    lp_ds = g["lnprob"]
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        time2 = thetime.time()
        print('MCMC run took ' + str((time2 - time1)/60.) + ' min')
        print('')
        print("Mean acceptance: "
            + str(np.mean(sampler.acceptance_fraction)))
        print('')

        if runmpi:
            pool.close()
        elif th > 1:
            sampler.pool.close()

        return sampler

if __name__ == '__main__':
    sampler = main(runmpi=True,nw=300,th=1,bi=1,fr=10000,use_hodlr=True)



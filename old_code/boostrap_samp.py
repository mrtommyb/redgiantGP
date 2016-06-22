import numpy as np
from numpy.random import choice
from ktransit import FitTransit
import h5py


def fitt_bootstrap(time, flux, ferr, resid, fitp, fits):
    T0, period, impact = fitp['pnum0']['T0'], fitp['pnum0']['period'], fitp['pnum0']['impact']
    rprs, ecosw, esinw = fitp['pnum0']['rprs'], fitp['pnum0']['ecosw'], fitp['pnum0']['esinw']
    zpt, rho, ld1, ld2 = fits['zpt'], fits['rho'], fits['ld1'], fits['ld2']
    fitT = FitTransit()
    fitT.add_guess_star(rho=rho,zpt=zpt,ld1=ld1,ld2=ld2)
    fitT.add_guess_planet(
        period=period, impact=impact,
        T0=T0, rprs=rprs,ecosw=ecosw,esinw=esinw)
    flux_resid = flux + choice(resid,size=len(flux))

    fitT.add_data(time=time, flux=flux_resid, ferr=ferr)
    vary_star = ['rho', 'zpt','ld1','ld2']      # free stellar parameters
    vary_planet = (['period',       # free planetary parameters
        'T0', 'impact',
        'rprs','ecosw','esinw'])
    fitT.free_parameters(vary_star, vary_planet)
    fitT.do_fit()                   # run the fitting
    return fitT.fitresult

fn = '/Users/tom/gitcode/GP_model_Kepler_data/data/koi2133_np1_priorTrue_dil0.0GP.hdf5'
f = h5py.File(fn)

time = f['time'][:61000]
flux = f['flux'][:61000]
ferr = f['err'][:61000]

fitT = FitTransit()
fitT.add_guess_star(rho=43.47/1000.,zpt=-5.68308831e-05,ld1=4.32157219e-01,ld2=2.69514828e-01)
fitT.add_guess_planet(
        period=6.2467332, impact=0.09,
        T0=1.36383739e+02, rprs=0.019429,ecosw=0.0,esinw=0.0)
fitT.add_data(time=time[:61000], flux=flux[:61000], ferr=ferr[:61000])
vary_star = ['rho', 'zpt','ld1','ld2']      # free stellar parameters
vary_planet = (['period',       # free planetary parameters
        'T0', 'impact',
        'rprs','ecosw','esinw'])
fitT.free_parameters(vary_star, vary_planet)
fitT.do_fit()                   # run the fitting

fitT.print_results()
initial_fitp = fitT.fitresultplanets
initial_fits = fitT.fitresultstellar
initial_resid = flux - fitT.transitmodel

npars = 10
nsamps = 10
outmap = np.zeros([nsamps,npars])
for i in arange(nsamps):
    outmap[i] = fitt_bootstrap(time[:61000], flux[:61000], ferr[:61000],
        initial_resid, initial_fitp, initial_fits)


[print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(*x)) for x in outmap]
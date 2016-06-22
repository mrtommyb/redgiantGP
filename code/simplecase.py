from __future__ import division, print_function

import numpy as np 
import ktransit

def makeFakeData():
    # simulate some data
    cadence = 1./48.
    time = np.arange(0,80,cadence)
    rvtime = np.arange(0,100,50)

    M = ktransit.LCModel()
    M.add_star(
            rho=0.001,
            ld1=0.2,
            ld2=0.4, 
            ld3=0.0,
            ld4=0.0, 
            dil=0.0,
            zpt=0.0, 
            veloffset=10 # new keyword, the radial velocity zero-point offset in m/s   
            )
    M.add_planet(
            T0=1.0,     
            period=10.0,
            impact=0.1,
            rprs=0.05,  
            ecosw=0.0, 
            esinw=0.0,
            occ=0.0,  
            rvamp=100.) # radial velocity semi-amplitude in m/s

    M.add_data(
            time=time,   
            itime=np.zeros_like(time)+cadence
            )

    M.add_rv(time=rvtime,                                   # radial velocity observation timestamps
            itime=np.zeros_like(rvtime)+cadence    # integration time of each timestamp
            )   

    tmod = M.transitmodel
    rvmodel = M.rvmodelv

    return time, tmod, rvmod

if '__name__' == __main__:

    # simulate some data
    time, tmod, rvmod = makeFakeData()
    ferr = np.zeroslike(tmod) + 1.E-4
    rverr = np.zeroslike(rvmod) + 5.

    # add some white noise
    tmod * np








import numpy as np
import sys
import ktransit
import h5py
import matplotlib.pyplot as plt
import george
from george.kernels import RBFKernel

def bin_data(phi,flux,bins,model=None):
    phi = np.array(phi)
    flux = np.array(flux)
    phibin = []
    fluxbin = []
    stdbin = []
    for i in (bins*np.arange(len(phi)/bins))+(bins/2):
        if model == None:
            goodpoints = np.ones(len(flux),dtype=bool)
        else:
            goodpoints = flux-model < 3* np.std(flux-model)
        flux2 = flux[goodpoints]
        phi2 = phi[goodpoints]
        phibin.append(np.median(phi2[i-bins/2:i+bins/2]))
        fluxbin.append(np.median(flux2[i-bins/2:i+bins/2]))
        stdbin.append(np.std(flux2[i-bins/2:i+bins/2]))
    return np.array(phibin), np.array(fluxbin), np.array(stdbin) / np.sqrt(bins)

def get_qf(time,flux,epoch,period,transitmodel=None):
    date1 = (time - epoch) + 0.5*period
    phi1 = (((date1 / period) - np.floor(date1/period)) * period) - 0.5*period
    q1 = np.sort(phi1)
    f1 = (flux[np.argsort(phi1)]) * -1.E6
    if transitmodel is not None:
        m1 = (transitmodel[np.argsort(phi1)]) * -1.E6
        return q1,f1,m1
    else:
        return q1,f1

def get_qf_rv(time,flux,err,epoch,period,rvmodel=None):
    date1 = (time - epoch) + 0.5*period
    phi1 = (((date1 / period) - np.floor(date1/period)) * period) - 0.5*period
    q1 = np.sort(phi1)
    f1 = (flux[np.argsort(phi1)])
    e1 = err[np.argsort(phi1)]
    if rvmodel is not None:
        m1 = (rvmodel[np.argsort(phi1)])
        return q1,f1,e1,m1
    else:
        return q1,f1,e1

def get_many_samples(gp1,gp2,time,resid,ferr,nsamples=300):
    kernel = gp1**2 * RBFKernel(gp2)
    gp = george.GaussianProcess(kernel)

    slist = np.arange(len(time) // 1000)
    samples = np.zeros([nsamples,len(slist)*1000])
    for i in slist:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], ferr[:][section])
        samples[:,section] = gp.sample_conditional(
            resid[section],time[section],
            size=nsamples)
    return samples

def get_sample(gp1,gp2,time,resid,ferr):
    kernel = gp1**2 * RBFKernel(gp2)
    gp = george.GaussianProcess(kernel)

    slist = np.arange(len(time) // 1000)
    sample = np.zeros(len(slist)*1000)
    for i in slist:
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(time[section], ferr[:][section])
        sample[section] = gp.predict(
            resid[section],time[section])[0]
    return sample


def plot_transit_best(hf,ax,ylim=[9000,-1000],bins=900):
    with h5py.File(hf) as f:
        g = f['mcmc']['chain'][:]
        lnprob = f['mcmc']['lnprob'][:]
        mle_idx = np.unravel_index(lnprob.argmax(),
            lnprob.shape)
        mle = g[mle_idx]
        M = ktransit.LCModel()
        M.add_star(rho=mle[0],zpt=mle[1],ld1=mle[2],
            ld2=mle[3],veloffset=mle[4])
        M.add_planet(T0=mle[7],period=mle[8],impact=mle[9],
            rprs=mle[10],ecosw=mle[11],esinw=mle[12],
            rvamp=mle[13],
            occ=mle[14],ell=mle[15],alb=mle[16])
        M.add_data(time=f['time'][:])
        M.add_rv(rvtime=f['rvtime'][:])

        #resid = f['flux'][:] - M.transitmodel
        #sample = get_sample(
        #    mle[5],mle[6],M.time,resid,f['err'][:])
        #len_samp = len(sample)
        phi,ffold,fmod = get_qf(M.time,
            f['flux'][:],
            M.T0,
            M.period,
            transitmodel=M.transitmodel)

        ax.scatter(np.r_[phi,phi+M.period],
            np.r_[ffold,ffold],color='k',alpha=0.05,s=0.5)
        ax.plot(np.r_[phi,phi+M.period],
            np.r_[fmod,fmod],color='r')
        bq1,bf1,be1 = bin_data(phi,ffold,bins)
        ax.errorbar(np.r_[bq1,bq1+M.period],
            np.r_[bf1,bf1],yerr=np.r_[be1,be1],
            ls='',color='b')
        ax.set_xlim([-0.5*M.period,1.5*M.period])
        ax.set_ylim(ylim)
        ax.set_xlabel('Time from mid-transit (days)')
        ax.set_ylabel('Transit depth (ppm)')
        ax.minorticks_on()
    return ax

def plot_rv_best(hf,ax,ylim=[-420,420]):
    with h5py.File(hf) as f:
        g = f['mcmc']['chain'][:]
        lnprob = f['mcmc']['lnprob'][:]
        mle_idx = np.unravel_index(lnprob.argmax(),
            lnprob.shape)
        mle = g[mle_idx]
        M = ktransit.LCModel()
        M.add_star(rho=mle[0],zpt=mle[1],ld1=mle[2],
            ld2=mle[3],veloffset=mle[4])
        M.add_planet(T0=mle[7],period=mle[8],impact=mle[9],
            rprs=mle[10],ecosw=mle[11],esinw=mle[12],
            rvamp=mle[13],
            occ=mle[14],ell=mle[15],alb=mle[16])
        M.add_data(time=f['time'][:])
        M.add_rv(rvtime=f['rvtime'][:])
        tmod = M.transitmodel
        phi,ffold,efold,fmod = get_qf_rv(M.rvtime,
            f['rvval'][:] - mle[4],
            np.sqrt(f['rverr'][:]**2 + mle[-1]**2),
            M.T0,M.period,rvmodel=M.rvmodel- mle[4])

        M.add_rv(rvtime=np.arange(0,M.period,0.002))
        tmod = M.transitmodel
        phi2,fmod2,efol2 = get_qf_rv(M.rvtime,
            M.rvmodel-mle[4],
            M.rvmodel,
            M.T0,M.period)

        lphi2 = int(len(phi2) / 2.)
        ax.plot(np.r_[phi2,phi2+M.period],
            np.r_[fmod2,fmod2],color='r')


        ax.errorbar(np.r_[phi,phi+M.period],
            np.r_[ffold,ffold],
            yerr=np.r_[efold,efold],color='k',alpha=1,ls='',
            fmt='.')

        ax.set_xlim([-0.5*M.period,1.5*M.period])
        ax.set_ylim(ylim)
        ax.set_xlabel('Time from mid-transit (days)')
        ax.set_ylabel('Radial Velocity (m/s)')
        ax.minorticks_on()
    return ax

def plot_rv_best2(hf,ax,ylim=[-420,420]):
    with h5py.File(hf) as f:
        g = f['mcmc']['chain'][:]
        lnprob = f['mcmc']['lnprob'][:]
        mle_idx = np.unravel_index(lnprob.argmax(),
            lnprob.shape)
        mle = g[mle_idx]
        M = ktransit.LCModel()
        M.add_star(rho=mle[0],zpt=mle[1],ld1=mle[2],
            ld2=mle[3],veloffset=mle[4])
        M.add_planet(T0=mle[5],period=mle[6],impact=mle[7],
            rprs=mle[8],ecosw=mle[9],esinw=mle[10],
            rvamp=mle[11],
            occ=mle[12],ell=mle[13],alb=mle[14])
        M.add_data(time=f['time'][:])
        M.add_rv(rvtime=f['rvtime'][:])
        tmod = M.transitmodel
        phi,ffold,efold,fmod = get_qf_rv(M.rvtime,
            f['rvval'][:] - mle[4],
            np.sqrt(f['rverr'][:]**2 + mle[-1]**2),
            M.T0,M.period,rvmodel=M.rvmodel- mle[4])

        M.add_rv(rvtime=np.arange(0,M.period,0.002))
        tmod = M.transitmodel
        phi2,fmod2,efol2 = get_qf_rv(M.rvtime,
            M.rvmodel-mle[4],
            M.rvmodel,
            M.T0,M.period)

        lphi2 = int(len(phi2) / 2.)
        #ax.plot(phi2/M.period,fmod2,color='r')
        #ax.plot((phi2[0:lphi2]+M.period)/M.period,
        #    fmod2[0:lphi2],color='r',ls=':')
        #ax.plot((phi2[lphi2:]-M.period)/M.period,
        #    fmod2[lphi2:],color='r',ls=':')
        ax.plot(phi2,fmod2,color='r')
        ax.plot((phi2[0:lphi2]+M.period),
            fmod2[0:lphi2],color='r',ls=':')
        ax.plot((phi2[lphi2:]-M.period),
            fmod2[lphi2:],color='r',ls=':')

        # ax.errorbar(phi /M.period,
        #     ffold,
        #     yerr=efold,color='k',alpha=1,ls='')
        # ax.errorbar((np.r_[phi-M.period,phi+M.period]) / M.period,
        #     np.r_[ffold,ffold],yerr=np.r_[efold,efold],
        #     color='k',alpha=0.3,ls='')
        # ax.set_xlim([-1,1])

        ax.errorbar(phi,
            ffold,
            yerr=efold,color='k',alpha=1,ls='')
        ax.errorbar((np.r_[phi-M.period,phi+M.period]) ,
            np.r_[ffold,ffold],yerr=np.r_[efold,efold],
            color='k',alpha=0.3,ls='')
        #ax.set_xlim([-1,1])

        ax.set_ylim(ylim)
        ax.set_xlabel('Time from mid-transit (days)')
        ax.set_ylabel('Radial Velocity (m/s)')
        ax.minorticks_on()
    return ax

params = {'backend': 'png',
            'axes.linewidth': 2.5,
            'axes.labelsize': 24,
            'axes.font': 'sans-serif',
            'axes.fontweight' : 'bold',
            'text.fontsize': 12,
            'legend.fontsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'text.usetex': True,
            'font.family': 'sans-serif',
            'legend.markersize': 200}
plt.rcParams.update(params)


if __name__ == '__main__':
    hf = 'koi2133_np1_priorTrue_dil0.0GP.hdf5'
    fig, axes = plt.subplots(2, 1, figsize=[6,9])
    ax1 = plot_transit_best(hf,axes[0],ylim=[2000,-1500])
    #ax1 = plot_transit_best(hf,axes[0],ylim=[1100,-600])
    ax2 = plot_rv_best(hf,axes[1],ylim=[-220,220])
    plt.tight_layout()
    plt.savefig(
        '/Users/tom/Projects/koi2133/data/{0}.png'.format(
            hf[0:7].strip('.0').strip('.')))



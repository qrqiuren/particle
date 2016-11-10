# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:39:20 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
import matplotlib.pyplot as plt

from ulaarray import ULAArray

from signalyielder import SignalYielder

from caponsampler import CaponSampler
from flomsampler import FLOMSampler

from naivetracker import NaiveTracker
from pftracker import PFTracker

# Modes can be selected within 'singleshot', 'flom', 'alpha', 'gsnr' and 'nss'.
# 'singleshot' is the fastest usage of the main routine. Others are
# evaluations of algorithms
mode = 'singleshot'
nruns = 100


def sim(totaltime, totalss, nss, alpha, flomorder, gsnr, retorig,
        randtrack=False):
    """
    Perform one round of tracking simulation.

    This function performs one round of simulation.
    The tracking results will be returned as 4 trackers in sequence of
    (Capon-NoPF, FLOM-NoPF, Capon-PF, FLOM-PF).

    Parameters
    ----------
    totaltime : float
        Total time in seconds.
    totalss : int
        Total number of snapshots to perform.
    nss : int
        Number of snapshots in each time step.
    alpha : float
        Alpha parameter of S-alpha-S noise.
    flomorder : float
        Moment order (p) of FLOM
    gsnr : float
        Generalized signal to noise ratio.
    retorig : bool
        If true, ground truth and estimations of each tracker will be included
        in returned tuple, like (proc, rmse, ts_t, ts_doa, esti); otherwise,
        the return tuple will be (proc, rmse).
    randtrack : bool
        If true, the ground truth DOA track will have random noise on its
        velocity. Otherwise, the ground truth DOA will be a linear function.
        The value is False by default.

    Returns
    -------
    proc : (4,) ndarray
        PROC% of each trackers.
    rmse : (4,) ndarray
        Root mean square error of each trackers.
    ts_t : (nts,) ndarray
        Actual time in each time step. Only returns when `retts` is true.
    ts_doa : (nts,) ndarray
        Ground truth DOA of each timesteps. Only returns when `retts` is true.
    esti : (4, nts) ndarray
        Estimations of each trackers in each timestep. `nts` is the number of
        time steps. Only returns when `retts` is true.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    nts = totalss // nss    # Number of timesteps
    gamma = 10 ** (-gsnr / 10)  # Gamma parameter of S-alpha-S noise

    # Define time and DOA angle vectors
    t = np.linspace(0., totaltime, totalss)
    if not randtrack:
        doa = np.linspace(0.25 * np.pi, 0.75 * np.pi, totalss)
    else:
        dt = t[1] - t[0]
        doa = np.zeros(t.shape)
        vel = ((0.75 * np.pi) - (0.25 * np.pi)) / totalss / dt
        doa[0] = 0.25 * np.pi
        for ss in range(totalss - 1):
            doa[ss+1] = doa[ss] + np.random.normal(vel, 5.) * dt

    # Compute them for each timestep
    ts_t = t[(nss//2):totalss:nss]
    ts_doa = doa[(nss//2):totalss:nss]
    ts_dt = ts_t[1] - ts_t[0]

    # Define array layout
    ula = ULAArray(nsensors=5, wavelength=15, sensordist=7.5)

    # Define signal generator
    siggen = SignalYielder(ula, alpha, gamma)

    # Construct samplers
    capon = CaponSampler(nss, ula)
    flom = FLOMSampler(nss, ula, flomorder)

    # Construct trackers
    caponnaive = NaiveTracker(capon, 360)
    flomnaive = NaiveTracker(flom, 360)

    initvel = (ts_doa[1] - ts_doa[0]) / ts_dt
    initdoa = ts_doa[0] - initvel * ts_dt
    caponpf = PFTracker(capon, ts_dt, nparts=32, neff=32, initdoa=initdoa,
                        initvel=initvel, initstd=0.1, noisestd=5, expo=5.)
    flompf = PFTracker(flom, ts_dt, nparts=32, neff=32, initdoa=initdoa,
                       initvel=initvel, initstd=0.1, noisestd=5, expo=5.)

    # Perform simulation for each trackers
    trackers = [caponnaive, flomnaive, caponpf, flompf]
    esti = np.zeros((len(trackers), nts))
    proc = np.zeros((len(trackers),))
    rmse = np.zeros((len(trackers),))
    proc_thresh = 1 * np.pi / 180

    # Process in each time step
    for ts in range(nts):
        y = siggen.batchgen(doa[ts*nss:(ts+1)*nss])

        for i in range(len(trackers)):
            theta = trackers[i].timestep(y)
            esti[i][ts] = theta
            proc[i] += abs(theta - ts_doa[ts]) < proc_thresh
            rmse[i] += (theta - ts_doa[ts]) ** 2

    proc *= 100. / nts
    rmse = np.sqrt(rmse / nts)

    if retorig:
        return proc, rmse, ts_t, ts_doa, esti
    else:
        return proc, rmse

if mode == 'singleshot':
    import time

    # Perform simulation
    start = time.clock()
    proc, rmse, ts_t, ts_doa, esti = \
        sim(totaltime=1., totalss=1600, nss=32, alpha=1.6,
            flomorder=1.1, gsnr=-4, retorig=True)
    print('PROC = ', proc)
    print('RMSE = ', rmse)
    stop = time.clock()
    print('Time: %f s' % (stop - start))

    # Plotting tracking results
    trackfig, trax = plt.subplots()
    trax.plot(ts_t, 180 / np.pi * ts_doa, color='k', label='Ground truth',
              linewidth=3)
    trax.plot(ts_t, 180 / np.pi * esti[0], 'b+-.', label='Capon')
    trax.plot(ts_t, 180 / np.pi * esti[1], 'r*--', label='FLOM')
    trax.plot(ts_t, 180 / np.pi * esti[2], 'g^-', label='Capon-PF')
    trax.plot(ts_t, 180 / np.pi * esti[3], 'md-', label='FLOM-PF')
    trax.legend(loc='best')
    trax.set_ylim(0, 180)
    trax.grid(axis='y')
    trax.set_xlabel('time')
    trax.set_ylabel('DOA estimation (degree)')

    # Plotting residual
    resfig, resax = plt.subplots()
    resax.plot([ts_t[0], ts_t[-1]], [0, 0], color='k', linewidth=3)
    resax.plot(ts_t, 180 / np.pi * (esti[0] - ts_doa), 'b+-.', label='Capon')
    resax.plot(ts_t, 180 / np.pi * (esti[1] - ts_doa), 'r*--', label='FLOM')
    resax.plot(ts_t, 180 / np.pi * (esti[2] - ts_doa), 'g^-', label='Capon-PF')
    resax.plot(ts_t, 180 / np.pi * (esti[3] - ts_doa), 'md-', label='FLOM-PF')
    resax.legend(loc='best')
    resax.set_ylim(-10, 10)
    resax.grid(axis='y')
    resax.set_xlabel('time')
    resax.set_ylabel('residual (degree)')

    plt.show()

elif mode == 'flom':
    # Different FLOM order
    flomorders = np.arange(0.5, 2.1, 0.1)
    proc = np.zeros((4, len(flomorders)))
    rmse = np.zeros((4, len(flomorders)))
    for k in range(nruns):
        for i in range(len(flomorders)):
            print('%d th run, FLOM order = %.1f' % (k, flomorders[i]))
            tmpproc, tmprmse = \
                sim(totaltime=1., totalss=1600, nss=32, alpha=1.6,
                    flomorder=flomorders[i], gsnr=-4, retorig=False)
            proc[:, i] += tmpproc
            rmse[:, i] += tmprmse
    proc /= nruns
    rmse /= nruns

    fig, ax = plt.subplots()
    ax.set_xlabel('FLOM order')
    ax.set_ylabel('PROC %')
    ax.plot(flomorders, proc[0], 'b+-.', label='Capon')
    ax.plot(flomorders, proc[1], 'r*--', label='FLOM')
    ax.plot(flomorders, proc[2], 'g^-', label='Capon-PF')
    ax.plot(flomorders, proc[3], 'md-', label='FLOM-PF')
    ax.set_ylim(0, 100)
    ax.legend(loc='best')

    fig, ax = plt.subplots()
    ax.set_xlabel('FLOM order')
    ax.set_ylabel('RMSE')
    ax.plot(flomorders, rmse[0], 'b+-.', label='Capon')
    ax.plot(flomorders, rmse[1], 'r*--', label='FLOM')
    ax.plot(flomorders, rmse[2], 'g^-', label='Capon-PF')
    ax.plot(flomorders, rmse[3], 'md-', label='FLOM-PF')
    ax.legend(loc='best')

    plt.show()

elif mode == 'alpha':
    # Different alpha
    alphas = np.arange(1.0, 2.1, 0.1)
    proc = np.zeros((4, len(alphas)))
    rmse = np.zeros((4, len(alphas)))
    for k in range(nruns):
        for i in range(len(alphas)):
            print('%d th run, alpha = %.1f' % (k, alphas[i]))
            tmpproc, tmprmse = \
                sim(totaltime=1., totalss=1600, nss=32, alpha=alphas[i],
                    flomorder=1.4, gsnr=-4, retorig=False)
            proc[:, i] += tmpproc
            rmse[:, i] += tmprmse
    proc /= nruns
    rmse /= nruns

    fig, ax = plt.subplots()
    ax.set_xlabel('alpha')
    ax.set_ylabel('PROC %')
    ax.plot(alphas, proc[0], 'b+-.', label='Capon')
    ax.plot(alphas, proc[1], 'r*--', label='FLOM')
    ax.plot(alphas, proc[2], 'g^-', label='Capon-PF')
    ax.plot(alphas, proc[3], 'md-', label='FLOM-PF')
    ax.set_ylim(0, 100)
    ax.legend(loc='best')

    fig, ax = plt.subplots()
    ax.set_xlabel('alpha')
    ax.set_ylabel('RMSE')
    ax.plot(alphas, rmse[0], 'b+-.', label='Capon')
    ax.plot(alphas, rmse[1], 'r*--', label='FLOM')
    ax.plot(alphas, rmse[2], 'g^-', label='Capon-PF')
    ax.plot(alphas, rmse[3], 'md-', label='FLOM-PF')
    ax.legend(loc='best')

    plt.show()

elif mode == 'gsnr':
    # Different alpha
    gsnrs = np.arange(-10., 11., 2.)
    proc = np.zeros((4, len(gsnrs)))
    rmse = np.zeros((4, len(gsnrs)))
    for k in range(nruns):
        for i in range(len(gsnrs)):
            print('%d th run, GSNR = %.1f' % (k, gsnrs[i]))
            tmpproc, tmprmse = \
                sim(totaltime=1., totalss=1600, nss=32, alpha=1.6,
                    flomorder=1.1, gsnr=gsnrs[i], retorig=False)
            proc[:, i] += tmpproc
            rmse[:, i] += tmprmse
    proc /= nruns
    rmse /= nruns

    fig, ax = plt.subplots()
    ax.set_xlabel('GSNR')
    ax.set_ylabel('PROC %')
    ax.plot(gsnrs, proc[0], 'b+-.', label='Capon')
    ax.plot(gsnrs, proc[1], 'r*--', label='FLOM')
    ax.plot(gsnrs, proc[2], 'g^-', label='Capon-PF')
    ax.plot(gsnrs, proc[3], 'md-', label='FLOM-PF')
    ax.set_ylim(0, 100)
    ax.legend(loc='best')

    fig, ax = plt.subplots()
    ax.set_xlabel('GSNR')
    ax.set_ylabel('RMSE')
    ax.plot(gsnrs, rmse[0], 'b+-.', label='Capon')
    ax.plot(gsnrs, rmse[1], 'r*--', label='FLOM')
    ax.plot(gsnrs, rmse[2], 'g^-', label='Capon-PF')
    ax.plot(gsnrs, rmse[3], 'md-', label='FLOM-PF')
    ax.legend(loc='best')

    plt.show()

elif mode == 'nss':
    # Different alpha
    nssbase = np.arange(2, 11)
    nss = 2 ** nssbase
    proc = np.zeros((4, len(nss)))
    rmse = np.zeros((4, len(nss)))
    for k in range(nruns):
        for i in range(len(nss)):
            print('%d th run, NSS = %d' % (k, nss[i]))
            tmpproc, tmprmse = \
                sim(totaltime=1., totalss=nss[i]*50, nss=nss[i], alpha=1.6,
                    flomorder=1.1, gsnr=-4, retorig=False)
            proc[:, i] += tmpproc
            rmse[:, i] += tmprmse
    proc /= nruns
    rmse /= nruns

    fig, ax = plt.subplots()
    ax.set_xlabel('Snapshots')
    ax.set_ylabel('PROC %')
    ax.plot(nssbase, proc[0], 'b+-.', label='Capon')
    ax.plot(nssbase, proc[1], 'r*--', label='FLOM')
    ax.plot(nssbase, proc[2], 'g^-', label='Capon-PF')
    ax.plot(nssbase, proc[3], 'md-', label='FLOM-PF')
    ax.set_ylim(0, 100)
    ax.legend(loc='best')

    fig, ax = plt.subplots()
    ax.set_xlabel('Snapshots')
    ax.set_ylabel('RMSE')
    ax.plot(nssbase, rmse[0], 'b+-.', label='Capon')
    ax.plot(nssbase, rmse[1], 'r*--', label='FLOM')
    ax.plot(nssbase, rmse[2], 'g^-', label='Capon-PF')
    ax.plot(nssbase, rmse[3], 'md-', label='FLOM-PF')
    ax.legend(loc='best')

    plt.show()

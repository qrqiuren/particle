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

totalss = 1000          # Number of total snapshots
nss = 20                # Number of snapshots in a timestep
nts = totalss // nss    # Number of timesteps

alpha = 1.6             # Alpha parameter of S-alpha-S noise
flomorder = 1.2         # Order of FLOM

GSNR = -8                   # Generalized SNR in dB
gamma = 10 ** (-GSNR / 10)  # Gamma parameter of S-alpha-S noise

# Define time and DOA angle vectors
t = np.linspace(0., 1., totalss)
doa = np.linspace(np.pi * 0.25, np.pi * 0.75, 1000)
dt = t[1] - t[0]
doa = np.zeros(t.shape)
vel = ((0.75 * np.pi) - (0.25 * np.pi)) / totalss / dt
doa[0] = 0.25 * np.pi
for i in range(totalss - 1):
    doa[i+1] = doa[i] + (vel + np.random.normal(0., 5.)) * dt

# Compute them for each timestep
ts_t = t[0:totalss:nss]
ts_doa = doa[0:totalss:nss]

# Define array layout
ula = ULAArray(nsensors=8, wavelength=3.2, sensordist=1.6)

# Define signal generator
siggen = SignalYielder(ula, alpha, gamma)

capon = CaponSampler(nss, ula)
flom = FLOMSampler(nss, ula, flomorder)

caponnaive = NaiveTracker(capon, 360)
flomnaive = NaiveTracker(flom, 360)

caponpf = PFTracker(capon, dt, nparts=32, neff=24, initdoa=doa[0], initvel=0.,
                    initstd=0.1, noisestd=10000, expo=5)
flompf = PFTracker(flom, dt, nparts=32, neff=24, initdoa=doa[0], initvel=0.,
                   initstd=0.1, noisestd=10000, expo=5)

trackers = [caponnaive, flomnaive, caponpf, flompf]
esti = dict((tracker, np.zeros(nts,)) for tracker in trackers)

# Process in each time step
for ts in range(nts):
    y = siggen.batchgen(doa[ts*nss:(ts+1)*nss])

    for tracker in trackers:
        esti[tracker][ts] = tracker.timestep(y)

# Plotting results
plt.plot(t, 180 / np.pi * doa, color='k', label='Ground truth')
plt.plot(ts_t, 180 / np.pi * esti[caponnaive], 'b+--', label='Capon')
plt.plot(ts_t, 180 / np.pi * esti[flomnaive], 'r*--', label='FLOM')
plt.plot(ts_t, 180 / np.pi * esti[caponpf], 'g^-', label='Capon-PF')
plt.plot(ts_t, 180 / np.pi * esti[flompf], 'md-', label='FLOM-PF')
plt.legend(loc='best')
plt.show()

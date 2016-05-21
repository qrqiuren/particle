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

order = 1.6             # Order of the noise
flomorder = 1.2         # ORder of FLOM

# Define time and DOA angle vectors
t = np.linspace(0., 1., totalss)
doa = np.linspace(np.pi * 0.25, np.pi * 0.75, 1000)
dt = t[1] - t[0]

# Define array layout
ula = ULAArray(nsensors=8, wavelength=3.2, sensordist=1.6)

# Define signal generator
siggen = SignalYielder(ula, order, 2.0)

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

for tracker in trackers:
    plt.plot(180 / np.pi * esti[tracker])
plt.show()

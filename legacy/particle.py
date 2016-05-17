# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:56:44 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>

This is a simulation program for acoustic signal tracking.

Reference:

[1] Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
"""

import time

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

from capon import doaSolver_Capon
from pf import doaSolver_PF

from salphas import salphas_cplx

# Arguments
sampling_rate = 44100       # Sampling rate
total_time = 0.5            # Total time
M = 3                       # Number of array elements
N = 90                      # Number of samples in a time slice
bf_resolution = 90          # Number of sample points of beamforming angle
wavelength = 3.             # Wavelength of the wave
d = 1.414                   # Sensor separation of ULA

dt = 1. / sampling_rate     # Interval of each time step
t = np.arange(0., total_time, 1. / sampling_rate)   # Time array
t.resize((1, t.size))

# TODO: Randomize signal
amplitude = 0.5 * np.ones(t.shape)          # Amplitude in each time step
phase = 2 * np.pi * np.random.rand(t.size)  # Random phase in each time step
signal = amplitude * np.exp(1j * phase)     # Signal in each time step

# TODO: Randomize DOA angle
theta_t = pi / 2 / total_time * t + pi / 4        # Circular movement
m = np.arange(M).reshape((M, 1))
steer = np.exp(-2j * pi * d / wavelength * m.dot(np.cos(theta_t)))  # Steering vector

y = np.zeros((M, t.size), dtype=np.complex)
for i in range(t.size):
    y[:, i] = steer[:, i] * signal[0, i]

# Randomize noise
noise = salphas_cplx(1.23, 1.5, size=(M, t.size))
y += noise

# Capon DOA estimation
start_clock = time.clock()
theta, estimated_Capon, Capon = doaSolver_Capon(y, wavelength, d, N, bf_resolution)
stop_clock = time.clock()
print('Time (Capon): %f s' % (stop_clock - start_clock))

"""
# Plot the Capon beamformer result of the first slice
plt.plot(180 / pi * theta[0], Capon[:, 0], 'o-')
plt.grid(True)
plt.title('Spatial Spectrum of Capon Estimator (DOA = 45 deg)')
plt.xlabel('Direction (degree)')
plt.ylabel('Power')
plt.show()
"""

# PF-FLOM DOA estimation
start_clock = time.clock()
init_status = np.array([pi / 4, 0.])
estimated_PF = doaSolver_PF(y, dt, 5., 1.99, init_status, .8, 10000000, wavelength,
                            d, 48, 42, N, bf_resolution)
stop_clock = time.clock()
print('Time (PF-FLOM): %f s' % (stop_clock - start_clock))

# Compare the difference of real and estimated
plt.plot(t.ravel(), 180. / pi * theta_t.ravel(), 'k')
time_slices = np.arange(t.size // N) * N / sampling_rate
plt.plot(time_slices, estimated_Capon / pi * 180., 'o:r')
plt.plot(time_slices, abs(estimated_PF) / pi * 180., '*-b')
plt.legend(['Ground Truth', 'Capon', 'PF-FLOM'], loc='best')
plt.title('DOA Tracking in S-alpha-S Noise Environment')
plt.xlabel('Time (s)')
plt.ylabel('DOA (deg)')
plt.show()

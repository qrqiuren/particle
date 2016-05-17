# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:09:37 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
from numpy.linalg import inv
from math import pi


def doaSolver_Capon(Y, wavelength, sensor_dist,
                    slice_size=8, bf_resolution=90):
    """
    Capon beamformer.

    Parameters
    ----------
    Y : (M, timesteps) ndarray
        The original data received from ULA. M is the number of sensors in ULA.
    wavelength : float
        The wavelength in meters.
    sensor_dist : float
        Distance between closest sensors.
    slice_size : int
        Number of samples in each slice.
    bf_resolution : int
        Resolution of beamformer. It decides how many samples will be in range
        [0, pi) in the algorithm output.

    Returns
    -------
    angle : (1, bf_resolution) ndarray
        An array with divisioned angle. The shape is (1, bf_resolution).
    estimated : (slice_count,) ndarray
        Estimated DOA of each time slice. values are in range [0, pi).
    capon : (bf_resolution, timesteps)
        Original spatial spectra calculated by Capon beamformer.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    M, timesteps = Y.shape

    slice_count = int(timesteps // slice_size)

    estimated = np.zeros((slice_count,))
    Capon = np.zeros((bf_resolution, slice_count))

    angle = np.arange(0., pi, pi / bf_resolution)
    angle.resize((1, bf_resolution))
    m = np.arange(M).reshape((M, 1))
    steer = np.exp(-2j * pi * sensor_dist / wavelength * m.dot(np.cos(angle))) # Steering vector of each angle
    for k in range(slice_count):
        y_slice = Y[:, k*slice_size:(k+1)*slice_size]
        r_hat = np.dot(y_slice, y_slice.T.conj()) / slice_size

        for th in range(bf_resolution):
            a = steer[:, th].reshape((M, 1))
            Capon[th, k] = 1. / np.dot(np.dot(a.T.conj(), inv(r_hat)), a).real
        estimated[k] = np.argmax(Capon[:, k]) * pi / bf_resolution

    return angle, estimated, Capon

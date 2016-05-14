# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:09:42 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
from numpy.linalg import inv
from math import pi


def doaSolver_PF(Y, dt, expo, flomexp, init_status, init_stdev, noise_stdev,
                 wavelength, sensor_dist, npart, nthr, slice_size,
                 bf_resolution):
    """
    DOA tracker on PF approach.

    Parameters
    ----------
    Y : (M, nts) ndarray
        The original data received from ULA. M is the number of sensors in ULA.
        nts is the number of time steps.
    dt : float
        The interval of each time step.
    expo : float
        The exponential exponent ksi. Should be positive. Read the reference
        for further details.
    flomexpo : float
        The FLOM exponent p. It should be 1 < p < alpha <= 2. Read the
        reference for further details.
    init_status : (2,) ndarray
        The initial status. init_status[0] is the initial DOA while
        init_status[1] is the initial velocity.
    init_stdev : float
        The standard deviation of initial particles.
    noise_stdev : float
        Standard deviation of noise.
    wavelength : float
        The wavelength in meters.
    sensor_dist : float
        Distance between closest sensors.
    npart : int
        This variable decides how many particles should be drawn.
    nthr : int
        The threshold of efficient particles. Should less than npart.
    slice_size : int
        Number of samples in each time slice.
    bf_resolution : int
        Resolution of beamformer. It decides how many samples will be in range
        [0, pi) in the algorithm output.

    Return
    ------
    estimated : (nslice,) ndarray
        Estimated DOA by this method.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    M, nts = Y.shape
    nslice = nts // slice_size

    A = np.array([[1, dt], [0, 1]])
    B = np.array([dt * dt / 2, dt])

    # angle = np.arange(0., pi, pi / bf_resolution)
    # angle.resize((1, bf_resolution))
    m = np.arange(M).reshape((M, 1))

    estimated = np.zeros((nslice,))

    # Draw particles and set initial weight
    # TODO: randomize particles with covariance, though covariance is less
    #       useful in real applications.
    x = np.tile(init_status, (npart, 1)).T + \
        np.random.normal(0., init_stdev, (2, npart))
    w = np.ones((npart,)) / npart               # Weight
    for k in range(nslice):
        # Draw particles in this timestep
        for l in range(npart):
            x[:, l] = A.dot(x[:, l]) + B * np.random.normal(0., noise_stdev)

        # Compute FLOM matrix
        Yk = Y[:, k*slice_size:(k+1)*slice_size]
        flom = Yk.dot((np.abs(Yk.T)**(flomexp-2)) * Yk.T.conj())

        for l in range(npart):
            # Compute the response of spatial spectra and likelihood
            steer = np.exp(-2j * pi * sensor_dist / wavelength *
                           np.cos(x[0, l]) * m)
            response = 1. / steer.T.conj().dot(inv(flom).dot(steer))
            likelihood = np.abs(response) ** expo

            # Compute the importance weight
            w[l] *= likelihood

        # Normalize weight
        w /= np.sum(w)

        # Resampling
        if 1. / np.sum(w ** 2) < nthr:
            cum = np.zeros((npart+1,))
            for l in range(npart):
                cum[l+1] = cum[l] + w[l]

            xprev = x
            x = np.zeros(xprev.shape)

            for l in range(npart):
                # Perform binary search
                key = np.random.rand()
                left = 0
                right = npart
                while right - left > 1:
                    mid = (left + right) // 2
                    if cum[mid] <= key:
                        left = mid
                    elif cum[mid] > key:
                        right = mid
                x[:, l] = xprev[:, left]

            w = np.ones((npart,)) / npart

        # Estimation
        estimated[k] = np.sum(w * x[0])

    return estimated

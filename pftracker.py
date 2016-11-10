# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:12:57 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
from numpy.random import normal

from tracker import Tracker


class PFTracker(Tracker):
    """
    Particle filtering DOA tracker.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, smp, dt, nparts, neff, initdoa, initvel, initstd,
                 noisestd, expo):
        """
        Parameters
        ----------
        smp : SpectrumSampler
            Sampler of spatial spectrum.
        dt : float
            Time interval of time steps.
        nparts : int
            Number of particles to be drawn by the particle filter.
        neff : int
            The threshold of efficient particles. Should less than `nparts`.
        initdoa : float
            Initial DOA angle.
        initvel : float
            Initial velocity of `initdoa`.
        initstd : float
            Standard deviation of initial particles.
        noisestd : float
            Standard deviation of noise.
        expo : float
            Exponent of computing likelihood of each particles.
        """
        self.smp = smp

        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([dt * dt / 2, dt])

        self.nparts = nparts
        self.neff = neff

        self.noisestd = noisestd

        self.expo = expo

        # Draw particles and set initial weight
        x0 = np.array([initdoa, initvel])
        self.x = np.tile(x0, (nparts, 1)).T + \
            (normal(0., initstd, (nparts, 2)) * self.B).T
        self.w = np.ones((nparts,)) / nparts

    def timestep(self, y):
        """
        Estimates DOA of incoming observations.

        Parameters
        ----------
        y : (nsensors, nss) ndarray
            Observations. `nsensor` is the number of sensors in the sensor
            array `sarr`. `nss` is the number of snapshots in the timestep.

        Returns
        -------
        angle : float
            Estimated angle of current time step.
        """
        x = self.x
        w = self.w
        A = self.A
        B = self.B

        # Compute covariance matrix
        smp = self.smp
        smp.compCov(y)

        # Draw particles and calculate weight for the current time step
        for i in range(self.nparts):
            x[:, i] = A.dot(x[:, i]) + normal(0., self.noisestd) * B
            response = smp.compSpecSample(x[0, i])
            likelihood = np.abs(response) ** self.expo
            w[i] *= likelihood

        w /= np.sum(w)

        # Wrap DOA angle to [0, pi)
        too_small = x[0, :] < 0
        too_large = x[0, :] >= np.pi
        while np.any(too_small) or np.any(too_large):
            x[0, too_small] = -x[0, too_small]
            x[0, too_large] = np.pi - x[0, too_large]
            too_small = x[0, :] < 0
            too_large = x[0, :] >= np.pi

        # Resampling
        if 1. / np.sum(w ** 2) < self.neff:
            nparts = self.nparts

            cum = np.zeros((nparts+1,))
            for i in range(nparts):
                cum[i+1] = cum[i] + w[i]

            xprev = x
            x = np.zeros(xprev.shape)

            for i in range(nparts):
                # Draw a random prob and perform binary search
                key = np.random.rand()
                left = 0
                right = nparts
                while right - left > 1:
                    mid = (left + right) // 2
                    if cum[mid] <= key:
                        left = mid
                    elif cum[mid] > key:
                        right = mid
                x[:, i] = xprev[:, left]

            w = np.ones((nparts,)) / nparts

        # Estimation
        angle = np.sum(w * x[0])
        return angle

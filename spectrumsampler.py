# -*- coding: utf-8 -*-
"""
Created on Sun May 15 01:47:46 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""


class SpectrumSampler:
    """
    Spectrum sampler base class.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, nss, sarr):
        """
        Parameters
        ----------
        nss : int
            Number of snapshots per time step.
        sarr : DOAArray
            Instance of sensor array.
        """
        self.nss = nss
        self.sarr = sarr

    def compCov(self, y):
        """
        Computes and restores the estimated covariance matrix of incoming
        observations.

        Parameters
        ----------
        y : (nsensors, nss) ndarray
            Observations. `nsensor` is the number of sensors in the sensor
            array `sarr`. `nss` is the number of snapshots in the timestep.

        Returns
        -------
        r : ndarray
            Estimated covariance matrix of `y`.
        """
        raise NotImplementedError

    def compSpecSample(self, angle):
        """
        Computes the spectrum in a sample.

        Parameters
        ----------
        angle : float
            Direction-of-arrival (DOA) angle in range [0, pi).

        Returns
        -------
        p : float
            The response of spacial spectrum in `angle`.
        """
        raise NotImplementedError

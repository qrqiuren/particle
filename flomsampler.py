# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:26:09 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

from numpy.linalg import inv

from spectrumsampler import SpectrumSampler


class FLOMSampler(SpectrumSampler):
    """
    Sampler of spatial spectrum calculated by FLOM matrix.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, nss, sarr, order):
        """
        Parameters
        ----------
        nss : int
            Number of snapshots per time step.
        sarr : DOAArray
            Instance of sensor array.
        order : float
            The order of statistical moment used to compute FLOM. Should be
            a positive number less than 2.
        """
        self.nss = nss
        self.sarr = sarr
        self.order = order

    def compCov(self, y):
        """
        Computes and restores the FLOM matrix of incoming observations before
        computing spatial spectrum.

        Parameters
        ----------
        y : (nsensors, nss) ndarray
            Observations. `nsensor` is the number of sensors in the sensor
            array `sarr`. `nss` is the number of snapshots in the timestep.

        Returns
        -------
        gamma : ndarray
            Estimated FLOM matrix of `y`.
        """
        self.gamma = y.dot(abs(y.T) ** (self.order - 2) * y.T.conj())
        return self.gamma

    def compSpecSample(self, angle):
        """
        Computes the spatial spectrum response in a specified angle with FLOM
        matrix.

        MUST compute the FLOM matrix (call `compFLOM()`) before calling this
        function.

        Parameters
        ----------
        angle : float
            Direction-of-arrival (DOA) angle in range [0, pi).

        Returns
        -------
        p : float
            The response of spatial spectrum.
        """
        a = self.sarr.steer(angle)
        p = 1. / (a.T.conj().dot(inv(self.gamma).dot(a)))
        return p

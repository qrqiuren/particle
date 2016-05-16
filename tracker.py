# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:14:49 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""


class Tracker:
    """
    DOA tracker base class.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, smp):
        """
        Parameters
        ----------
        smp : SpectrumSampler
            Sampler of spatial spectrum.
        """
        self.smp = smp

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
        raise NotImplementedError

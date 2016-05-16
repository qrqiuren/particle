# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:31:03 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np

from tracker import Tracker


class NaiveTracker(Tracker):
    """
    DOA tracker without any filtering. It always computes the full spectrum.

    Attributes
    ----------
    angles : (nangles,) ndarray
        Vector of angles to be sampled in a spatial spectrum. It is sampled in
        a linear space.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, smp, nangles):
        """
        Parameters
        ----------
        smp : SpectrumSampler
            Sampler of spatial spectrum.
        nangles : int
            Number of angle samples in a spatial spectrum. The angles will be
            sampled as a linear space in [0, pi).
        """
        self.smp = smp
        self.nangles = nangles
        self.angles = np.linspace(0, np.pi, nangles, endpoint=False)

    def timestep(self, y, retspec=False):
        """
        Estimates DOA of incoming observations.

        Parameters
        ----------
        y : (nsensors, nss) ndarray
            Observations. `nsensor` is the number of sensors in the sensor
            array `sarr`. `nss` is the number of snapshots in the timestep.
        retspec : bool
            If this is true, it returns a tuple (`angle`, `spec`). Otherwise,
            return `angle` only.

        Returns
        -------
        angle : float
            Estimated angle of current time step.
        spec : (nangles,) ndarray
            The full spectrum of current time step. It only returns when
            `retspec` is true.
        """
        smp = self.smp
        angles = self.angles

        smp.compCov(y)
        spec = np.zeros(angles.shape)
        for i in range(self.nangles):
            spec[i] = smp.compSpecSample(angles[i])

        angle = angles[np.argmax(spec)]

        return angle, spec

# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:50:09 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""


class DOAArray:
    """
    Base class of sensor array in DOA problems.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """

    def __init__(self, nsensor, wavelength):
        """
        Initialization of DOASteer.

        Parameters
        ----------
        nsensor : int
            Number of sensors.
        wavelength : float
            Wavelength of incoming wave.
        """
        self.nsensor = nsensor
        self.wavelength = wavelength

    def steer(self, angle):
        """
        Returns the steering vector of a DOA angle.

        You should implement it by every kind of array layout.

        Parameters
        ----------
        angle : float
            DOA angle in [0, pi).

        Returns
        -------
        a : (nsensor,) ndarray
            The steering vector corresponding to `angle`.
        """
        raise NotImplementedError

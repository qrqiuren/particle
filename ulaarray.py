# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:57:49 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
from numpy import pi, cos

from doaarray import DOAArray


class ULAArray(DOAArray):
    """
    ULA sensor array class.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """

    def __init__(self, nsensors, wavelength, sensordist):
        """
        Parameters
        ----------
        nsensors : int
            Number of sensors.
        wavelength : float
            Wavelength of incoming wave.
        sensordist : float
            Distance between nearest sensors.
        """
        self.nsensors = nsensors
        self.wavelength = wavelength
        self.sensordist = sensordist

    def steer(self, angle):
        """
        Returns the steering vector of a DOA angle.

        Parameters
        ----------
        angle : float
            DOA angle in [0, pi).

        Returns
        -------
        a : (nsensors,) ndarray
            The steering vector corresponding to `angle`.
        """
        nsensors = self.nsensors
        sensordist = self.sensordist
        wavelength = self.wavelength

        m = np.arange(nsensors).reshape((nsensors,))
        a = np.exp(-2j * pi * sensordist / wavelength * cos(angle) * m)
        return a

if __name__ == '__main__':
    arr = ULAArray(4, 3., 1.5)
    print(arr.steer(0.))

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:01:31 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np

from salphas import salphas_cplx


class SignalYielder:
    """
    Generating and yielding signals.

    See the reference for definition and form of signal.

    Reference
    ---------
    Zhong, X., Prekumar, A. B., and Madhukumar, A. S., "Particle filtering for
    acoustic source tracking in impulsive noise with alpha-stable process",
    IEEE Sensors Journal, Feb. 2013, Vol. 13 No. 2: 589-600.
    """
    def __init__(self, sarr, alpha, gamma):
        """
        Parameters
        ----------
        sarr : DOAArray
            Instance of sensor array.
        alpha : float
            Alpha coefficient (characteristic exponent) of S-alpha-S
            distribution.
        gamma : float
            Gamma coefficient (dispersion parameter) of S-alpha-S distribution.
        """
        self.sarr = sarr
        self.alpha = alpha
        self.gamma = gamma

    def gen(self, angle):
        """
        Generate a snapshot of signal under a given DOA angle.

        The amplitude is 1 and phase follows uniform distribution of
        [0, 2*pi).

        Parameters
        ----------
        angle : float
            Angle of current snapshot.

        Returns
        -------
        y : (nsensors,)
            A snapshot in DOA `y`. `nsensors` is the number of sensors in
            `sarr`.
        """
        a = self.sarr.steer(angle)
        phase = np.random.random() * np.pi * 2
        signal = np.exp(1j * np.pi * phase)

        y = signal * a
        noise = salphas_cplx(self.alpha, self.gamma, size=y.shape)
        y += noise

        return y

    def batchgen(self, angles):
        """
        Generate snapshots of signal under a given DOA angle.

        The amplitude is 1 and phase follows uniform distribution of
        [0, 2*pi).

        Parameters
        ----------
        angles : (nsnapshots,)
            Angle of current snapshot.

        Returns
        -------
        y : (nsensors, nsnapshots)
            A snapshot in DOA `y`. `nsensors` is the number of sensors in
            `sarr`.
        """
        nsnapshots = angles.size

        phase = np.random.random(nsnapshots) * np.pi * 2
        y = salphas_cplx(self.alpha, self.gamma,
                         size=(self.sarr.nsensors, nsnapshots))

        for i in range(nsnapshots):
            a = self.sarr.steer(angles[i])
            signal = np.exp(1j * np.pi * phase[i])
            y[:, i] += signal * a

        return y

if __name__ == '__main__':
    from ulaarray import ULAArray

    ula = ULAArray(5, 3, 1.5)
    sgen = SignalYielder(ula)

    y = sgen.gen(np.pi / 2)
    print(y)

    y = sgen.batchgen(np.linspace(0., np.pi, 7))
    print(y)

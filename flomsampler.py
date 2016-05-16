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

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from ulaarray import ULAArray
    from salphas import salphas_cplx

    nsensors = 10
    nss = 20
    groundtruth = 68    # Ground truth

    ula = ULAArray(nsensors, 3, 1.5)
    flom = FLOMSampler(nss, ula, 1.1)

    t, step = np.linspace(0., 1., nss, retstep=True)
    phase = 2 * np.pi * np.random.rand(t.size)
    signal = 0.8879 * np.exp(1j * np.pi * phase)
    a = ula.steer(np.pi * groundtruth / 180)

    y = np.zeros((nsensors, t.size), dtype=np.complex)
    for i in range(t.size):
        y[:, i] = a * signal[i]
    noise = 2 * salphas_cplx(1.1, 2, size=(nsensors, t.size))
    y += noise

    flom.compCov(y)

    angles = np.linspace(0., np.pi, 360)
    spec = np.zeros(angles.shape)
    for i in range(angles.size):
        spec[i] = flom.compSpecSample(angles[i])

    plt.plot(angles / np.pi * 180, spec)
    plt.plot([groundtruth, groundtruth], [0, 1], 'k')
    plt.title('FLOM beamformer (Ground truth = %d deg)' % groundtruth)
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sun May 15 02:20:24 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

from numpy.linalg import inv

from spectrumsampler import SpectrumSampler


class CaponSampler(SpectrumSampler):
    """
    Sampler of Capon beamforming estimator's spatial spectrum.

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
        self.r = y.dot(y.T.conj()) / self.nss
        return self.r

    def compSpecSample(self, angle):
        """
        Computes the spectrum in a sample with Capon beamformer.

        MUST compute the covariance matrix (call `compCov()`) before calling
        this function.

        Parameters
        ----------
        angle : float
            Direction-of-arrival (DOA) angle in range [0, pi).

        Returns
        -------
        p : float
            The response of Capon beamformer.
        """
        a = self.sarr.steer(angle)
        p = 1. / (a.T.conj().dot(inv(self.r).dot(a)))
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
    capon = CaponSampler(nss, ula)

    t, step = np.linspace(0., 1., nss, retstep=True)
    phase = 2 * np.pi * np.random.rand(t.size)
    signal = 0.8879 * np.exp(1j * np.pi * phase)
    a = ula.steer(np.pi * groundtruth / 180)

    y = np.zeros((nsensors, t.size), dtype=np.complex)
    for i in range(t.size):
        y[:, i] = a * signal[i]
    noise = 0.5 * salphas_cplx(1.1, 2, size=(nsensors, t.size))
    y += noise

    capon.compCov(y)

    angles = np.linspace(0., np.pi, 360)
    spec = np.zeros(angles.shape)
    for i in range(angles.size):
        spec[i] = capon.compSpecSample(angles[i])

    plt.plot(angles / np.pi * 180, spec)
    plt.plot([groundtruth, groundtruth], [0, 1], 'k')
    plt.title('Capon beamformer (Ground truth = %d deg)' % groundtruth)
    plt.show()

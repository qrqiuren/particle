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

        # Discard imaginary part
        return p.real

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from ulaarray import ULAArray
    from salphas import salphas_cplx
    from caponsampler import CaponSampler

    nsensors = 10
    nss = 20
    nangles = 360

    groundtruth = 68    # Ground truth in degrees

    expo = 5

    ula = ULAArray(nsensors, 3, 1.5)
    capon = CaponSampler(nss, ula)
    flom = FLOMSampler(nss, ula, 1.1)

    t, step = np.linspace(0., 1., nss, retstep=True)
    phase = 2 * np.pi * np.random.rand(t.size)
    signal = np.exp(1j * np.pi * phase)
    a = ula.steer(np.pi * groundtruth / 180)

    y = np.zeros((nsensors, t.size), dtype=np.complex)
    for i in range(t.size):
        y[:, i] = a * signal[i]
    noise = salphas_cplx(1.5, 3.0, size=(nsensors, t.size))
    y += noise

    capon.compCov(y)
    flom.compCov(y)

    angles = np.linspace(0., np.pi, nangles)
    caponspec = np.zeros(angles.shape)
    flomspec = np.zeros(angles.shape)
    caponexpospec = np.zeros(angles.shape)
    flomexpospec = np.zeros(angles.shape)
    for i in range(angles.size):
        caponspec[i] = capon.compSpecSample(angles[i])
        flomspec[i] = flom.compSpecSample(angles[i])
        caponexpospec[i] = capon.compSpecSample(angles[i]) ** expo
        flomexpospec[i] = flom.compSpecSample(angles[i]) ** expo
#    caponspec /= caponspec[groundtruth * nangles // 180]
#    flomspec /= flomspec[groundtruth * nangles // 180]
#    caponexpospec /= caponexpospec[groundtruth * nangles // 180]
#    flomexpospec /= flomexpospec[groundtruth * nangles // 180]
    caponspec /= np.max(caponspec)
    flomspec /= np.max(flomspec)
    caponexpospec /= np.max(caponexpospec)
    flomexpospec /= np.max(flomexpospec)

    plt.plot(angles / np.pi * 180, caponspec, 'k:', label='Capon')
    plt.plot(angles / np.pi * 180, flomspec, 'k--', label='FLOM')
    plt.plot(angles / np.pi * 180, caponexpospec, 'k-.',
             label=r'Capon, $\xi$=5')
    plt.plot(angles / np.pi * 180, flomexpospec, 'k-',
             label=r'FLOM, $\xi$=5')
    plt.plot([groundtruth, groundtruth], [0, 1.2], 'k')
    plt.xlabel('DOA angle (degree)')
    plt.ylabel('normalized amplitude')
    plt.title('Spatial spectrum (Ground truth = %d deg)' % groundtruth)
    plt.legend()
    plt.show()

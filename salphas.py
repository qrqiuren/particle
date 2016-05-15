# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 21:03:48 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

import numpy as np
from numpy import random
from numpy import sin, cos, tan, arctan, log, pi


def salphas(alpha, gamma, beta=0., size=None):
    """
    Generate random variables under S-alpha-S distribution.

    Please check the reference paper for furthur details on algorithms and
    symbols.

    Parameters
    ----------
    alpha : float
        Alpha coefficient (characteristic exponent) of S-alpha-S distribution.
    gamma : float
        Gamma coefficient (dispersion parameter) of S-alpha-S distribution.
    beta : float
        Beta coefficient (skewness parameter) of alpha stable distribution. By
        default, this value will be 0 as the definition of S-alpha-S
        distribution. But it allows configuration to generate samples in a
        broader situation.
    size : tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. If not indicated, a single value will be returned.

    Returns
    -------
    a : float
        A real number or a real matrix with size `size` which is the sample of
        the distribution.

    Reference
    ---------
    Tsakalides, P., and Nikias C. L., "The robust covariation-based MUSIC
    (ROC-MUSIC) algorithm for bearing estimation in impulsive noise
    environments", IEEE Transactions on Signal Processing,
    Jul. 1996, Vol. 44 No. 7: 1623-1633
    """
    # Draw random vars
    pi2 = pi / 2
    W = random.exponential(scale=1., size=size)
    U = random.uniform(low=-pi2, high=pi2, size=size)

    # Sampling with params alpha and beta
    if alpha == 1:
        p2bu = pi2 + beta * U
        S = (p2bu * tan(U) - beta * log(pi2 * W * cos(U) / p2bu)) / pi2
    else:
        U0 = -pi2 * beta * (1 - abs(1 - alpha)) / alpha
        auu0 = alpha * (U - U0)
        D = (cos(arctan(beta * tan(pi2 * alpha)))) ** (1 / alpha)
        E = sin(auu0) / ((cos(U)) ** (1 / alpha))
        F = (cos(U - auu0) / W) ** ((1 - alpha) / alpha)
        S = D * E * F

    # Making gamma efficient
    a = gamma ** (1 / alpha) * S
    return a


def salphas_cplx(alpha, gamma, size=None):
    """
    Generate complex random variables under S-alpha-S distribution.

    Please check the reference paper for furthur details on algorithms and
    symbols.

    Parameters
    ----------
    alpha : float
        Alpha coefficient (characteristic exponent) of S-alpha-S distribution.
    gamma : float
        Gamma coefficient (dispersion parameter) of S-alpha-S distribution.
    size : tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. If not indicated, a complex value will be returned.

    Returns
    -------
    a : float
        A real number or a real matrix with size `size` which is the sample of
        the distribution.

    Reference
    ---------
    Tsakalides, P., and Nikias C. L., "The robust covariation-based MUSIC
    (ROC-MUSIC) algorithm for bearing estimation in impulsive noise
    environments", IEEE Transactions on Signal Processing,
    Jul. 1996, Vol. 44 No. 7: 1623-1633
    """
    # Generate sample of S-alpha-S random variable A and calc its square root
    agamma = cos(pi * alpha / 4) ** 2
    a = salphas(alpha=alpha, beta=1, gamma=agamma, size=size)

    # Generate Gaussian sample G1 and G2
    sigma = 2 * (gamma ** (1 / alpha))
    g1 = random.normal(0., sigma, size=size)
    g2 = random.normal(0., sigma, size=size)

    # Calculate the final sample
    x = (np.array(a, dtype=np.complex) ** 0.5) * (g1 + 1j * g2)
    return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    noisefig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    g = random.normal(0., 1., size=(1000,))
    x = salphas(alpha=1.23, beta=0., gamma=1., size=(1000,))
    ax1.plot(x, color='b')
    ax1.set_title('1000 random S-alpha-S samples (alpha=1.23, gamma=1)')
    ax2.plot(g, color='r')
    ax2.set_title('1000 random standard Gaussian samples')
    plt.show()

    histfig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.hist(x, 100, range=(-20, 20), color='b')
    ax1.set_title('Histogram of above S-alpha-S samples')
    ax2.hist(g, 100, range=(-20, 20), color='r')
    ax2.set_title('Histogram of above standard Gaussian samples')

    cplxfig, (ax1, ax2) = plt.subplots(nrows=2)
    x = salphas_cplx(alpha=1.23, gamma=1., size=(1000,))
    ax1.plot(x.real, color='b')
    ax1.set_title('Real and imag part of complex S-alpha-S samples')
    ax2.plot(x.imag, color='g')
    plt.show()

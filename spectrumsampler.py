# -*- coding: utf-8 -*-
"""
Created on Sun May 15 01:47:46 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""


class SpectrumSampler:
    """
    """
    def compSpecSample(self, angle):
        """
        Computes the spectrum in a sample.

        Parameters
        ----------
        angle : float
            Direction-of-arrival (DOA) angle in range [0, pi).

        Returns
        -------
        p : float
            The response of spacial spectrum in `angle`.
        """
        raise NotImplementedError

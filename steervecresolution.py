# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:43:53 2016

@author: Huang Hongye <qrqiuren@users.noreply.github.com>
"""

# Computes resolution of steering vectors

import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0., np.pi, 360)
res4 = 4 * 3 / 2 * np.pi * np.sin(theta)
res8 = 8 * 7 / 2 * np.pi * np.sin(theta)
res12 = 12 * 11 / 2 * np.pi * np.sin(theta)

plt.plot(180 / np.pi * theta, res4, 'k-', label='M=4')
plt.plot(180 / np.pi * theta, res8, 'k-.', label='M=8')
plt.plot(180 / np.pi * theta, res12, 'k--', label='M=12')
plt.legend(loc='best')
plt.xlabel('DOA')
plt.ylabel('Resolution')
plt.show()

import numpy as np
import h5py
from scipy.fft import ifftn, fftn, fftfreq
from scipy.ndimage import fourier_gaussian
from scipy.stats import binned_statistic_dd
import os, sys

class Particles:
    def __init__(self, ngrid, L):
        self.mesh = np.zeros((ngrid, ngrid, ngrid))
        self.ngrid = ngrid
        self.h = L/ngrid
        self.range = [[0,L],[0,L],[0,L]]

    def assign(self, pos, value):
        if isinstance(value, float):
            ret = binned_statistic_dd(pos, value*pos.shape[0], self.ngrid, range=self.range)
        elif isinstance(value, np.ndarray) & (np.ndim(value) == 3):
            ret = binned_statistic_dd(pos, value, self.ngrid, range=self.range)
        else:
            raise TypeError('value must be a float scalar or 3D numpy array')
        self.mesh = ret.statistic / self.h**3
    
    def tensor(self, Rs, soft=1e-38):
        k = fftfreq(self.ngrid, d=self.h)[np.meshgrid[0:self.ngrid,0:self.ngrid,0:self.ngrid]]
        Ghat = -1. / (np.sum(k**2, axis=0) + soft)
        Ghat[0,0,0] = 0.
        FFTphi = fourier_gaussian(fftn(self.mesh), sigma=Rs) * Ghat
        Hij = -np.einsum('iklm,jklm->ijklm', k, k)
        return ifftn(FFTphi * Hij, axes=(2,3,4)).real

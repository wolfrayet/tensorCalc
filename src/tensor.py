import numpy as np
import h5py
from scipy.fft import ifftn, fftn, fftfreq
from scipy.ndimage import fourier_gaussian
from scipy.stats import binned_statistic_dd
from scipy.interpolate import interpn
import os, sys

class Particles:
    def __init__(self, ngrid, L):
        self.rho = np.zeros((ngrid, ngrid, ngrid))
        self.phi = None
        self.FFTphi = None
        self.tensor = None
        self.coords = None

        self.ngrid = ngrid
        self.h = L/ngrid
        self.range = [[0,L],[0,L],[0,L]]
        self.k = fftfreq(ngrid, d=self.h)[np.mgrid[0:ngrid,0:ngrid,0:ngrid]] * np.pi * 2

    def assign(self, pos, value):
        if isinstance(value, float):
            ret = binned_statistic_dd(pos, value*pos.shape[0], self.ngrid, range=self.range)
        elif isinstance(value, np.ndarray) & (np.ndim(value) == 1):
            ret = binned_statistic_dd(pos, value, statistic='sum', bins=self.ngrid, range=self.range)
        else:
            raise TypeError('value must be a float scalar or 1D numpy array')
        self.rho = ret.statistic / self.h**3
        coords = ret.bin_edges[0][:-1] + ret.bin_edges[0][1:]
        self.coords = (coords, coords, coords)
    
    def FFTpotential(self, Rs, soft=1e-38, workers=4):
        Ghat = -1. / (np.sum(self.k**2, axis=0) + soft)
        Ghat[0,0,0] = 0.
        self.FFTphi = fourier_gaussian(fftn(self.rho, workers=workers), sigma=Rs) * Ghat

    def PotentialField(self, Rs=1, soft=1e-38, workers=4, update=False, pos=None, value=None):
        if self.FFTphi is None:
            self.FFTpotential(Rs, soft=soft, workers=workers)
        if update:
            if pos is not None and value is not None:
                self.assign(pos, value)
                self.FFTpotential(Rs, soft=soft, workers=workers)
            else:
                raise TypeError('pos and value must not be none to update the grids')
        self.phi = ifftn(self.FFTphi, workers=workers).real

    def PotentialInterp(self, pos):
        if self.phi is None:
            raise TypeError('phi is none')
        if self.coords is None:
            raise TypeError('coords is none')
        return interpn(self.coords, self.phi, pos)
    
    def TensorField(self, Rs=1, soft=1e-38, workers=4, update=False, pos=None, value=None):
        '''
        tensor format:
            i j index
            0 0 0
            1 0 1
            1 1 2
            2 0 3
            2 1 4
            2 2 5
        '''
        if self.FFTphi is None:
            self.FFTpotential(Rs, soft=soft, workers=workers)
        if update:
            if pos is not None and value is not None:
                self.assign(pos, value)
                self.FFTpotential(Rs, soft=soft, workers=workers)
            else:
                raise TypeError('pos and value must not be none to update the grids')
        # Hij = -np.einsum('iklm,jklm->ijklm', self.k, self.k)
        # self.tensor = ifftn(self.FFTphi * Hij, axes=(2,3,4), workers=workers).real
        tensor = np.zeros((6, self.ngrid, self.ngrid, self.ngrid))
        for i in range(3):
            for j in range(i+1):
                index = int((i**2+i)/2 + j)
                tensor[index] = ifftn(self.FFTphi*self.k[i]*self.k[j], workers=workers).real
        self.tensor = tensor

    def TensorInterp(self, pos):
        '''
        interpolate tensor at given positions (pos) from the field
        '''
        if self.tensor is None:
            raise TypeError('tensor is none')
        if self.coords is None:
            raise TypeError('coords is none')

        tensor = np.zeros((pos.shape[0], 6))
        for i in range(6):
            tensor[i] = interpn(self.coords, self.tensor[i], pos)
        return tensor

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError(f'Use: python {sys.argv[0]} [file]')
    
    if not os.path.isfile(sys.argv[1]):
        raise FileNotFoundError

    # parameters
    ngrid = 256                 # number of grids
    Lbox = 500                  # box size, ckpc/h
    Rs = 1                      # smoothing scale (FWHM), ckpc/h
    workers = os.cpu_count()    # number of CPU for FFT parallel computing

    # read data
    hf = h5py.File(sys.argv[1], 'r')['particles']
    pos = np.stack((hf['x'], hf['y'], hf['z']), axis=1)
    mass = hf['host_lgmass']
    del hf

    # assign grid
    grid = Particles(ngrid, Lbox)
    grid.assign(pos, mass)

    # calculate tensor field
    grid.TensorField(Rs=Rs, workers=workers)

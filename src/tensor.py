import numpy as np
import h5py
from scipy.fft import ifftn, fftn, fftfreq
from scipy.ndimage import fourier_gaussian
from scipy.stats import binned_statistic_dd
from scipy.interpolate import interpn
import os, sys

def interp_task(coord, pos, value, i):
    return interpn(coord, value[i], pos)
class Particles:
    def __init__(self, ngrid, L):
        '''
        ngrid:  number of grids per side
        L:      box size in comoving Mpc/h
        '''
        # grid level
        self.rho = np.zeros((ngrid, ngrid, ngrid))
        self.rho_smooth = None
        self.FFTrho = None
        self.phi = None
        self.FFTphi = None
        self.tensor = None
        self.coords = None

        # parameters
        self.ngrid = ngrid
        self.h = L/ngrid
        self.range = [[-L/2,L/2],[-L/2,L/2],[-L/2,L/2]]
        self.k = fftfreq(ngrid, d=self.h)[np.mgrid[0:ngrid,0:ngrid,0:ngrid]] * np.pi * 2

    def assign(self, pos, mass):
        '''
        Assign patticles to the grid and create a grid of mass density.

        pos:    particle position in comoving Mpc/h
        mass:   particle mass
        '''
        if isinstance(mass, float):
            ret = binned_statistic_dd(pos, mass*pos.shape[0], statistic='sum', bins=self.ngrid, range=self.range)
        elif isinstance(mass, np.ndarray) & (np.ndim(mass) == 1):
            ret = binned_statistic_dd(pos, mass, statistic='sum', bins=self.ngrid, range=self.range)
        else:
            raise TypeError('mass must be a float scalar or 1D numpy array')
        # self.rho = ret.statistic / self.h**3
        self.rho = ret.statistic / np.mean(ret.statistic) - 1
        coords = 0.5*(ret.bin_edges[0][:-1] + ret.bin_edges[0][1:])
        self.coords = (coords, coords, coords)

    def smooth(self, Rs, workers=4, update=False):
        if self.FFTrho is None or update:
            self.FFTrho = fourier_gaussian(fftn(self.rho, workers=workers), sigma=Rs/self.h)
        self.rho_smooth = ifftn(self.FFTrho, workers=workers).real
        return

    def DensInterp(self, pos):
        if self.rho_smooth is None:
            raise TypeError('must smooth the density field first')
        if self.coords is None:
            raise TypeError('coords is None')
        return interpn(self.coords, self.rho_smooth, pos, bounds_error=False, fill_value=None)
    
    def FFTpotential(self, Rs, soft=1e-38, workers=4, update=True):
        '''
        Calculate the potential field in FFT.

        Rs:         smoothing length (FWHM) in comoving Mpc/h
        soft:       softening
        workers:    CPU for FFT parallel computing
        '''
        if self.FFTrho is None or update:
            self.smooth(Rs, workers=workers)
        
        Ghat = -1. / (np.sum(self.k**2, axis=0) + soft)
        Ghat[0,0,0] = 0.
        self.FFTphi = self.FFTrho * Ghat
        return

    def PotentialField(self, Rs=1, soft=1e-38, workers=4, update=False, pos=None, mass=None):
        '''
        The function will calculate the potential field if FFTphi is empty. Users can also update
        position and mass of particles by passing update=True.

        Rs:         smoothing length (FWHM) in comoving Mpc/h
        soft:       softening
        workers:    CPU for FFT parallel computing
        '''
        if self.FFTphi is None:
            self.FFTpotential(Rs, soft=soft, workers=workers)
        if update:
            if pos is not None and mass is not None:
                self.assign(pos, mass)
                self.FFTpotential(Rs, soft=soft, workers=workers)
            elif Rs is not None:
                self.FFTpotential(Rs, soft=soft, workers=workers)
            else:
                raise TypeError('pos and mass must not be none to update the grids')
        self.phi = ifftn(self.FFTphi, workers=workers).real

    def PotentialInterp(self, pos):
        '''
        Interpolate potential at given position(s) from the grid level.

        pos:    position for interpolation in comoving Mpc/h
        '''
        if self.phi is None:
            raise TypeError('phi is none')
        if self.coords is None:
            raise TypeError('coords is none')
        return interpn(self.coords, self.phi, pos)
    
    def TensorField(self, Rs=1, soft=1e-38, workers=4, update=False, pos=None, mass=None):
        '''
        The function will calculate the potential field if FFTphi is empty. Users can also update
        position and mass of particles by passing update=True.

        Rs:         smoothing length (FWHM) in comoving Mpc/h
        soft:       softening
        workers:    CPU for FFT parallel computing

        tensor Tij is stored as a flatten array with indexing as following:
            i j index
            0 0 0
            1 0 1
            1 1 2
            2 0 3
            2 1 4
            2 2 5
            index = (i*i + i)/2 + j
        '''
        if self.FFTphi is None:
            self.FFTpotential(Rs, soft=soft, workers=workers)
        if update:
            if pos is not None and mass is not None:
                self.assign(pos, mass)
                self.FFTpotential(Rs, soft=soft, workers=workers)
            elif Rs is not None:
                self.FFTpotential(Rs, soft=soft, workers=workers)
            else:
                raise TypeError('pos and mass must not be none to update the grids')
        
        self.tensor = np.zeros((6, self.ngrid, self.ngrid, self.ngrid))       
        
        for i in range(3):
            for j in range(i+1):
                index = int((i*i + i)/2 + j)
                self.tensor[index] = ifftn(-self.FFTphi*self.k[i]*self.k[j], workers=workers).real
    
    def TensorInterp(self, pos):
        '''
        Interpolate tensor at given position(s) from the grid level.

        pos:    position for interpolation in comoving Mpc/h
        '''
        if self.tensor is None:
            raise TypeError('tensor is none')
        if self.coords is None:
            raise TypeError('coords is none')

        tensor = np.zeros((pos.shape[0], 6))
        for i in range(6):
            tensor[:,i] = interpn(self.coords, self.tensor[i], pos, bounds_error=False, fill_value=None)
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

    # interpolate
    x = np.array([10, 20])
    y = np.array([300, 300])
    z = np.array([60, 60])
    xyz = np.stack((x, y, z), axis=1)
    tensor = grid.TensorInterp(xyz)

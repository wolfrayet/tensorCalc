import warnings
import illustris_python as il
import numpy as np
from numpy import floor, pi
from scipy.fft import fftfreq, ifftn, fftn
from scipy.stats import binned_statistic_dd
from scipy.ndimage import generic_filter
import numba
import glob
import h5py

def cic(pos, value, grid, h):
    if type(value) is float:
        value = np.ones(pos.shape[0]) * value
    
    if pos.shape[1] == 2:
        cic_2D(pos, value, grid, h)
        return
    elif pos.shape[1] == 3:
        cic_3D(pos, value, grid, h)
        return 
    else:
        warnings.warn('dimension should be 2 or 3', UserWarning)
        return

@numba.jit(nopython=True, parallel=True)
def cic_2D(pos, value, grid, h):
    N = grid.shape[0]
    grid[:] = 0.
    for m in numba.prange(pos.shape[0]):
        ix, iy = int(floor(pos[m,0]/h-0.5)), int(floor(pos[m,1]/h-0.5))
        ixp, iyp = (ix+1)%N, (iy+1)%N
        ux = (pos[m,0]/h - 0.5 - ix)
        uy = (pos[m,1]/h - 0.5 - iy)
        ix, iy = (ix+1)%N, (iy+1)%N
        
        grid[ix, iy] += (1-ux) * (1-uy) * value[m]
        grid[ix, iyp] += (1-ux) * uy * value[m]
        grid[ixp, iy] += ux * (1-uy) * value[m]
        grid[ixp, iyp] += ux * uy * value[m]
    grid /= h**2
    return

@numba.jit(nopython=True, parallel=True)
def cic_3D(pos, value, grid, h):
    N = grid.shape[0]
    grid[:] = 0.
    for m in numba.prange(pos.shape[0]):
        ix, iy, iz = int(floor(pos[m,0]/h-0.5)),\
            int(floor(pos[m,1]/h-0.5)),\
            int(floor(pos[m,2]/h-0.5))
        ixp, iyp, izp = (ix+1)%N, (iy+1)%N, (iz+1)%N
        ux = (pos[m,0]/h - 0.5 - ix)
        uy = (pos[m,1]/h - 0.5 - iy)
        uz = (pos[m,2]/h - 0.5 - iz)
        ix, iy, iz = (ix+1)%N, (iy+1)%N, (iz+1)%N
        
        grid[ix,   iy,  iz] += (1-ux) * (1-uy) * (1-uz) * value[m]
        grid[ix,  iyp,  iz] += (1-ux) * uy * (1-uz) * value[m]
        grid[ixp,  iy,  iz] += ux * (1-uy) * (1-uz) * value[m]
        grid[ixp, iyp,  iz] += ux * uy * (1-uz) * value[m]
        grid[ix,   iy, izp] += (1-ux) * (1-uy) * uz * value[m]
        grid[ixp,  iy, izp] += ux * (1-uy) * uz * value[m]
        grid[ix,  iyp, izp] += (1-ux) * uy * uz * value[m]
        grid[ixp, iyp, izp] += ux * uy * uz * value[m]
    grid /= h**3
    return

@numba.jit(nopython=True, parallel=True)
def cic_3D_potential(pos, value, grid, h):
    N = grid.shape[0]
    grid[:] = 0.
    num = np.zeros_like(grid)
    for m in numba.prange(pos.shape[0]):
        ix, iy, iz = int(floor(pos[m,0]/h-0.5)),\
            int(floor(pos[m,1]/h-0.5)),\
            int(floor(pos[m,2]/h-0.5))
        ixp, iyp, izp = (ix+1)%N, (iy+1)%N, (iz+1)%N
        ux = (pos[m,0]/h - 0.5 - ix)
        uy = (pos[m,1]/h - 0.5 - iy)
        uz = (pos[m,2]/h - 0.5 - iz)
        ix, iy, iz = (ix+1)%N, (iy+1)%N, (iz+1)%N
        
        grid[ix,   iy,  iz] += (1-ux) * (1-uy) * (1-uz) * value[m]
        grid[ix,  iyp,  iz] += (1-ux) * uy * (1-uz) * value[m]
        grid[ixp,  iy,  iz] += ux * (1-uy) * (1-uz) * value[m]
        grid[ixp, iyp,  iz] += ux * uy * (1-uz) * value[m]
        grid[ix,   iy, izp] += (1-ux) * (1-uy) * uz * value[m]
        grid[ixp,  iy, izp] += ux * (1-uy) * uz * value[m]
        grid[ix,  iyp, izp] += (1-ux) * uy * uz * value[m]
        grid[ixp, iyp, izp] += ux * uy * uz * value[m]

        num[ix,   iy,  iz] += (1-ux) * (1-uy) * (1-uz)
        num[ix,  iyp,  iz] += (1-ux) * uy * (1-uz)
        num[ixp,  iy,  iz] += ux * (1-uy) * (1-uz)
        num[ixp, iyp,  iz] += ux * uy * (1-uz)
        num[ixp,  iy, izp] += ux * (1-uy) * uz
        num[ix,  iyp, izp] += (1-ux) * uy * uz
        num[ix,   iy, izp] += (1-ux) * (1-uy) * uz
        num[ixp, iyp, izp] += ux * uy * uz
    return num

def get_particle(snap, basePath, particle_type, **kwargs):
    pos = il.snapshot.loadSubset(basePath, snap, particle_type, \
        ['Coordinates'], **kwargs)
    pos = pos / 1000.       # unit: Mpc / h
    return pos

def get_potential(snap, basePath, particle_type,**kwargs):
    x = il.snapshot.loadSubset(basePath, snap, particle_type, \
        ['Potential'], **kwargs)
    return x

def get_subhalo_pos(snap, basePath, subhaloID):
    pos = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloPos'])
    return pos[subhaloID]

def build_dm_rho_cube(snap, basePath, L, N, mass, method='cic',**kwargs):
    pos = get_particle(snap, basePath, 'dm', **kwargs)
    h = L / N
    if method == 'cic':
        rho = np.ones((N,N,N))
        cic(pos, mass, rho, h)
        return rho
    elif method == 'ngp':
        rho, _ = np.histogramdd(pos, N, range=[[0,L],[0,L],[0,L]])
        return rho
    else:
        warnings.warn(f'method {method} does not exist', UserWarning)

def build_dm_rho_cube_large_set(snap, basePath, L, N, mass, method='cic'):
    path = basePath+f'snapdir_{snap:03d}/'
    flist = glob.glob(path+'*.hdf5')
    h = L / N
    rho = np.zeros((N, N, N))
    if method == 'cic':
        for fname in flist:
            print(f'open {fname}')
            rho_grid = np.zeros((N, N, N))
            try:
                f = h5py.File(fname, 'r')
            except OSError as e:
                print(e)
            else:
                pos = f['PartType1']['Coordinates'][:] / 1000
                cic(pos, mass, rho_grid, h)
                rho += rho_grid
                del pos
            finally:
                f.close()
        return rho
    elif method == 'ngp':
        for fname in flist:
            print(f'open {fname}')
            with h5py.File(fname, 'r') as f:
                pos = f['PartType1']['Coordinates'][:] / 1000
                rho_grid, _ = np.histogramdd(pos, N, range=[[0,L],[0,L],[0,L]])
                rho += rho_grid
                del pos
        return rho
    else:
        warnings.warn(f'method {method} does not exist', UserWarning)

def build_phi_cube(snap, basePath, L, N, method='cic', **kwargs):
    pos = get_particle(snap, basePath, 'dm', **kwargs)
    phi = get_potential(snap, basePath, 'dm')
    h = L / N
    if method == 'cic':
        grid = np.ones((N,N,N))
        cic_3D_potential(pos, phi, grid, h)
        return grid
    elif method == 'ngp':
        grid, _, _ = binned_statistic_dd(pos, phi, \
            statistic='mean', bins=N, range=[[0,L],[0,L],[0,L]])
        return grid
    else:
        warnings.warn(f'method {method} does not exist', UserWarning)

def build_phi_cube_large_set(snap, basePath, L, N, method='cic'):
    path = basePath+f'snapdir_{snap:03d}/'
    flist = glob.glob(path+'*.hdf5')
    h = L / N
    phi_grid = np.zeros((N, N, N))
    phi_num = np.zeros((N, N, N))

    if method == 'cic':
        for fname in flist:
            print(f'open {fname}')
            try:
                f = h5py.File(fname, 'r')
            except OSError as e:
                print(e)
            else:
                pos = f['PartType1']['Coordinates'][:] / 1000
                phi = f['PartType1']['Potential'][:]
                grid = np.zeros((N, N, N))
                num = cic_3D_potential(pos, phi, grid, h)
                phi_grid += grid
                phi_num += num
                del pos, phi, grid, num
            finally:
                f.close()
    elif method == 'ngp':
        for fname in flist:
            with h5py.File(path+fname, 'r') as f:
                pos = f['PartType1']['Coordinates'][:] / 1000
                phi = f['PartType1']['Potential'][:]
                grid, _, _ = binned_statistic_dd(pos, phi, \
                    statistic='sum', bins=N, range=[[0,L],[0,L],[0,L]])
                num, _, _ = binned_statistic_dd(pos, phi, \
                    statistic='count', bins=N, range=[[0,L],[0,L],[0,L]])
                phi_grid += grid
                phi_num += num
            del pos, phi, grid, num
    else:
        warnings.warn(f'method {method} does not exist', UserWarning)
        return
    
    np.seterr(divide='ignore')
    mean_phi = phi_grid / phi_num
    nearest = generic_filter(mean_phi, np.nanmean, size=(3,3,3), mode='nearest')
    mean_phi[phi_num==0] = nearest[phi_num==0]
    return mean_phi

def cal_FFTfreq(N, h, dim=3):
    if dim == 2:
        k = fftfreq(N, d=h)[np.mgrid[0:N,0:N]] * pi * 2.
        k2 = k[0]**2 + k[1]**2
        return k, k2
    elif dim == 3:
        k = fftfreq(N, d=h)[np.mgrid[0:N,0:N,0:N]] * pi * 2.
        k2 = k[0]**2 + k[1]**2 + k[2]**2
        return k, k2
    else:
        warnings.warn('dimension should be 2 or 3', UserWarning)
        return

def cal_Wk(k2, Rs):
    # Wk = np.exp(-0.5 * Rs**2 * k2 / (2*np.pi))
    Wk = np.exp(-0.5 * Rs**2 * k2)
    return Wk #/ np.sum(Wk)

def cal_FFTphi(rho, Rs, k, k2):
    Ghat = -1. / (k2 + 1e-38)
    if k.shape[0] == 2:
        Ghat[0,0] = 0.
    elif k.shape[0] == 3:
        Ghat[0,0,0] = 0.
    else:
        warnings.warn('dimension should be 2 or 3', UserWarning)
        return
    Wk = cal_Wk(k2, Rs)
    return fftn(rho) * Ghat * Wk

def cal_phi(rho, Rs, k, k2):
    FFTphi = cal_FFTphi(rho, Rs, k, k2)
    return ifftn(FFTphi).real

def cal_tensor(FFTphi, k):
    if k.shape[0] == 2:
        Hij = -np.einsum('ikl,jkl->ijkl', k, k)
        tensor = ifftn(FFTphi * Hij, axes=(2,3)).real
        return tensor
    elif k.shape[0] == 3:
        Hij = -np.einsum('iklm,jklm->ijklm',k,k)
        tensor = ifftn(FFTphi * Hij, axes=(2,3,4)).real
        return tensor
    else:
        warnings.warn('dimension should be 2 or 3', UserWarning)

@numba.jit(nopython=True, parallel=True)
def interpolate(grid, pos, value, h):
    if grid.ndim == 2:
        N = grid.shape[0]
        value[:] = 0.
        for m in numba.prange(pos.shape[0]):
            ix, iy = int(floor(pos[m,0]/h-0.5)),\
                int(floor(pos[m,1]/h-0.5))
            ixp, iyp = (ix+1)%N, (iy+1)%N
            ux = (pos[m,0]/h - 0.5 - ix)
            uy = (pos[m,1]/h - 0.5 - iy)
            ix, iy = (ix+1)%N, (iy+1)%N

            value[m] += grid[ix, iy] * (1-ux) * (1-uy)
            value[m] += grid[ix, iyp] * (1-ux) * uy
            value[m] += grid[ixp, iy] * ux * (1-uy)
            value[m] += grid[ixp, iyp] * ux * uy
        # value *= h**2
    elif grid.ndim == 3:
        N = grid.shape[0]
        value[:] = 0.
        for m in numba.prange(pos.shape[0]):
            ix, iy, iz = int(floor(pos[m,0]/h-0.5)),\
                int(floor(pos[m,1]/h-0.5)),\
                int(floor(pos[m,2]/h-0.5))
            ixp, iyp, izp = (ix+1)%N, (iy+1)%N, (iz+1)%N
            ux = (pos[m,0]/h - 0.5 - ix)
            uy = (pos[m,1]/h - 0.5 - iy)
            uz = (pos[m,2]/h - 0.5 - iz)
            ix, iy, iz = (ix+1)%N, (iy+1)%N, (iz+1)%N
            
            value[m] += grid[ix,   iy,  iz] * (1-ux) * (1-uy) * (1-uz)
            value[m] += grid[ix,  iyp,  iz] * (1-ux) * uy * (1-uz)
            value[m] += grid[ixp,  iy,  iz] * ux * (1-uy) * (1-uz)
            value[m] += grid[ixp, iyp,  iz] * ux * uy * (1-uz)
            value[m] += grid[ix,   iy, izp] * (1-ux) * (1-uy) * uz
            value[m] += grid[ixp,  iy, izp] * ux * (1-uy) * uz
            value[m] += grid[ix,  iyp, izp] * (1-ux) * uy * uz
            value[m] += grid[ixp, iyp, izp] * ux * uy * uz
        # value *= h**3
    else:
        print('dimension must be 2 or 3')

@numba.jit(nopython=True, parallel=True)
def interpolate_tensor(tensor, pos, value, h):
    M = tensor.shape[0]
    if (value.ndim == 3) & (value.shape[1] == tensor.shape[0]):
        for i in numba.prange(M):
            for j in numba.prange(M):
                interpolate(tensor[i,j], pos, value[:,i,j], h)
    else:
        print('incorrect dimension')
            

# def build_den_cube(snap, basePath, boxsize, resol, **kwargs):
#     dm_pos = il.snapshot.loadSubset(basePath, snap, 'dm', ['Coordinates'], **kwargs)
#     dm_pos = dm_pos / 1000.     # unit: Mpc / h
    
#     den, _ = np.histogramdd(dm_pos, resol, range=[[0,boxsize],[0,boxsize],[0,boxsize]])
#     avgN_per_pix=np.float(len(dm_pos))/resol**3

#     return (den - avgN_per_pix) / avgN_per_pix
    
# def cal_Amp_FFTden(den, resol):
#     FFTden = np.fft.fftn(den)
#     Amp_FFTden = np.absolute(FFTden) / resol**3
#     Amp_FFTden[0,0,0] = 0

#     return Amp_FFTden

# def cal_freq_FFT(boxsize, resol):
#     dx = boxsize / np.double(resol)
#     fxyz = np.fft.fftfreq(resol, d=dx)[np.mgrid[0:resol,0:resol,0:resol]]
#     kxyz = fxyz * 2 * np.pi
#     # k2 = np.sqrt(fxyz[0]**2 + fxyz[1]**2 + fxyz[2]**2)
#     return kxyz

# def cal_Wk_FFT(kxyz, Rs):
#     k2 = np.abs(kxyz[0]**2 + kxyz[1]**2 + kxyz[2]**2)
#     Wk = np.exp(-0.5 * k2 * Rs**2)
#     return k2, Wk

# def cal_FFTphi(Amp_FFTden, Wk, k2):
#     FFTphi = Amp_FFTden / (k2 + 1e-38) * Wk
#     FFTphi[0,0,0] = 0
#     return FFTphi

# def cal_phi(Amp_FFTden, Wk, k2):
#     return np.fft.ifftn(cal_FFTphi(Amp_FFTden, Wk, k2))

# def cal_FFTphi_tensor(Amp_FFTden, Wk, kxyz, k2):
#     Amp_FFTtensor = Amp_FFTden * Wk / (k2 + 1e-38)
#     Amp_FFTtensor[0,0,0] = 0
#     Hessian_kk = np.einsum('iklm,jklm->ijklm', kxyz, kxyz)
#     FFTtensor = Amp_FFTtensor * Hessian_kk
#     return FFTtensor

# def cal_phi_grad(Amp_FFTden, Wk, kxyz, k2):
#     return np.fft.ifftn(cal_FFTphi_tensor(Amp_FFTden, Wk, kxyz, k2))

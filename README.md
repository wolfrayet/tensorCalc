# tensorCalc
This grid-based code calculates the mass density, the potential, and the tensor field as grid level from particle-based simulation data, and can further interpolates values (right now potential and tensor) at any given position. The code assigns particles into the grid via the Nearest-Grid Point (NGP) algorithm to create the matter density field. Then later it utilize FFT to calculate the potential and tensor efficeintly.  


The structure of the repository is as the following 

* `src`: source code
* `notebook`: example notebook or any other jupyter notebook
* `data`: any data goes into here

## Requirement
* `numpy`
* `scipy`
* `h5py`

## Data Description (copy from slack)
There are three datasets: `particles`, `halos`, and `mock_gals`. `Particles` and `halos` are all galaxies in the 500 Mpc/h cube and are identical, it's an artifact of how the density estimation code was written. The dataset `mock_gals` is a subset of particles that are within the SDSS reconstructed volume (`sdss_flag` = 1).
The galaxies (dataset particles) have these fields:

* X/Y/Z (in comoving Mpc/h)
* `host_lgmass`, the mass of the host halo in Msun/h
* `host_del8`, the matter overdensity measured in a 8 Mpc/h sphere centered on the host halo
* `cen_flag`, 0 for central, 1 for satellite
* `sdss_flag`, 0 is outside the SDSS region, 1 is inside

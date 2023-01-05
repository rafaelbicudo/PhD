# PhD

Repository with scripts using during my PhD.

## Dependencies

No library is needed for running bash scripts, while for Python scripts different packages are adopted. Complete usage of all tools requires the following packages:

* numpy
* argparse
* os
* subprocess
* pandas
* matplotlib
* sklearn
* scipy
* [DICEtools](https://github.com/hmcezar/dicetools)

## Description

A brief description of Python scripts can be obtained by running the `-h` flag, e.g., `python linear-regression.py -h` provides the input parameters and output files.

### w_g16 

Perform quantum mechanical calculations using the Gaussian16 package (as built in the *HPC - aguia4* cluster) to optimize the long-range parameter $(\omega)$. Specifically, it writes submission files to:

1. Create a directory for a given $\omega$. 
2. Optimize the geometry.
3. Compute the cationic and anionic energies at the ground-state (GS) optimized geometry.
4. Change the $\omega$ and go back to 1.

### pos_g16

Create a .csv file that provides, for each $\omega$ value, the following:

1. Anion energy at the GS optimized geometry - $E_{N-1}^0$
2. Cation energy at the GS optimized geometry - $E_{N+1}^0$
3. Neutral energy at the GS optimized geometry - $E_N^0$
4. Highest occupied molecular orbital energy at the neutral GS optimized geometry - $\epsilon_{HOMO}$
5. Lowest unoccupied molecular orbital energy at the neutral GS optimized geometry - $\epsilon_{LUMO}$

### linear-regression.py

Performs a linear regression to determine Ryckaert-Bellemans torsional coefficients used in the parameterization of classical dihedral energies. The optimization uses quantum mechanical calculations of a rigid scan via Gaussian09/16 code, performs a cubic interpolation in the quantum data and fit the coefficients to reproduce the polynomial. It recieves as input:

1. Gaussian09/16 output file - `gaussianlogfile`
2. DICEtools xyz rotations file - `xyzrotationsfile`
3. GROMACS topology file - `topfile`
4. Number of points (configurations) in the rigid scan - `npoints`
5. Atoms defining the torsional angle - `a1`, `a2`, `a3` and `a4`

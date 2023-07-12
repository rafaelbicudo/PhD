# PhD

Repository with scripts using during my PhD.

## Dependencies

No library is needed for running bash scripts, while for Python scripts different packages are adopted. Complete usage of all tools requires the following packages:

* numpy
* argparse
* os
* re
* subprocess
* pandas
* matplotlib
* sklearn
* scipy
* [DICEtools](https://github.com/hmcezar/dicetools)

## Description

A brief description of Python scripts can be obtained by running the `-h` flag, e.g., `python linear-regression.py -h` provides the input parameters and output files.

### w_g16 

Performs quantum mechanical calculations using the Gaussian16 package (as built in the *HPC - aguia4* cluster) to optimize the long-range parameter $(\omega)$. Specifically, it writes submission files to:

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

### evaporation.py

Interface with [GROMACS](https://www.gromacs.org/) developed for performing solvent evaporation. By specifying the amount of solvent and evaporation rate, the script removes solvent molecules and call GROMACS to perform NPT simulations in a loop. It recieves the following as input:

1. GROMACS molecular structure file (.gro) - `grofile`
2. GROMACS topology file (.top) - `topfile`
3. Evaporation rate (in percentage) - `evapRate`
4. Total amount of solvent to be removed (in percentage) - `evapTotal`
5. Cluster where calculations are going to be performed (lovelace or lince2) - `cluster`
6. Removes solvent molecules of the current configuration - `--dynamic`, `-d`

### LR.py

Performs a linear regression to determine Ryckaert-Bellemans torsional coefficients used in the parameterization of classical dihedral energies. The optimization uses quantum mechanical calculations of a rigid scan via Gaussian09/16 code, performs a cubic interpolation in the quantum data and fit the coefficients to reproduce the polynomial. It recieves the following as input:

1. Gaussian09/16 output file - `gaussianlogfile`
2. DICEtools xyz rotations file - `xyzrotationsfile`
3. GROMACS topology file (.itp) - `topfile`
4. DICE .txt file - `txtfile`
5. DICE .dfr file - `dfrfile`
6. Number of points (configurations) in the rigid scan - `npoints`
7. Atoms defining the torsional angle - `a1`, `a2`, `a3` and `a4`
8. Linear regression method (least-square, ridge or lasso) - `--method`, `-m`
9. Lasso/Ridge linear regression L1/L2 penalty coefficient - `--alpha`
10. Set weights for minimum points during the fit - `--weight`, `-w` [default=1]
11. Remove overlap configurations - `--remove-overlap`, `-r`
12. Cutoff for atomic overlap - `--cutoff`, `-c` [default = 0.5]
13. Clip dihedral energies to the provided value (in kcal/mol) [-value, value] - `--set-max-barrier`, `-b` [default = `None`]

### merge_traj.py

Combine all `cycle*-traj.gro` trajectory files from GROMACS into a single one. The goal is to visualize with VMD the evaporation as a single process.

1. Number of evaporation cycles - `ncycles`

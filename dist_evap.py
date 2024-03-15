#!/usr/bin/env python3

"""
Script to perform evaporation of solvent using GROMACS.

[X] Remove randomly a given percentage of solvent and create a new .gro file.
[X] Update the number of solvent molecules in the topology (.top) file.
[X] Run a NPT simulation.
[X] Make a loop for complete solvent elimination.

[ ] Change the code structure to run a single simulation, exploring the restart
utility from GROMACS.

OBS: Do not use a residue name with letters and numbers changing, ex: A1E2C5 -> BUG CORRECTED!
OBS: Do not have a topology file with underscore ("_") in the name.
OBS: Solute must be at the beginning of topology file.
OBS: Works only with one type of solvent (resname = 'SOL').

AUTHOR: Rafael Bicudo Ribeiro (@rafaelbicudo)
DATE: DEZ/2022
"""

import argparse
import numpy as np
import os
import subprocess as sp

Symbol_to_mass = {
    'H': 1.0078250, 'He': 4.00260, 'Li': 6.941, 'Be': 9.01218, 'B': 10.81, 'C': 12.000, 
    'N': 14.0030740, 'O': 15.9949146, 'F': 18.9984033, 'Ne': 20.179, 'Na': 22.98977, 'Mg': 24.305, 
    'Al': 26.98154, 'Si': 28.0855, 'P': 30.97376, 'S': 31.9720718, 'Cl': 35.453, 'Ar': 39.948, 
    'K': 39.0983, 'Ca': 40.08, 'Sc': 44.9559, 'Ti': 47.90, 'V': 50.9415, 'Cr': 51.996, 
    'Mn': 54.9380, 'Fe': 55.847, 'Ni': 58.9332, 'Co': 58.70, 'Cu': 63.546, 'Zn': 65.38, 
    'Ga': 69.72, 'Ge': 72.59, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.80, 
    'Tc': 98, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.4, 'Ag': 107.868, 'Cd': 112.41, 
    'In': 114.82, 'Sn': 118.69, 'Sb': 121.75, 'Te': 127.60, 'I': 126.9045, 'Xe': 131.30, 
    'Cs': 132.9054, 'Ba': 137.33, 'La': 138.9055, 'Ce': 140.12, 'Pr': 140.9077, 'Nd': 144.24, 
    'Pm': 145, 'Sm': 150.4, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.9254, 'Dy': 162.50, 
    'Ho': 164.9304, 'Er': 167.26, 'Tm': 168.9342, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 
    'Ta': 180.9479, 'W': 183.85, 'Re': 186.207, 'Os': 190.2, 'Ir': 192.22, 'Pt': 195.09, 
    'Au': 196.9665, 'Hg': 200.59, 'Tl': 204.37, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209, 
    'At': 210, 'Rn': 222, 'Fr': 223, 'Ra': 226.0254, 'Ac': 227.0278, 'Th': 232.0381, 
    'Pa': 231.0359, 'U': 238.029, 'Np': 237.0482, 'Pu': 242, 'Am': 243, 'Cm': 247, 
    'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257, 'Md': 258, 'No': 250, 
    'Lr': 260, 'Rf': 261, 'Db': 262, 'Sg': 263, 'Bh': 262, 'Hs': 255, 
    'Mt': 256, 'Ds': 269, 'Rg': 272, 'Cn': 277
}

def read_top_file(topfile: str) -> dict:
	"""
	Extract the molecules from topology (.top) file.

	PARAMETERS:
	topfile - the .top file from GROMACS with number of molecules.

	OUTPUT:
	top_data - nested dictionary -> {moltype: {'resname': str, 'amount': int}}
	"""

	# Create a nested dictionary with topology data
	top_data = {}	

	# Define a data index and parse the molecules in the topology file
	i = 0

	with open(topfile, "r") as top:
		line = top.readline()

		while "[ molecules ]" not in line:
			line = top.readline()

		while True:
			line = top.readline()
			words = line.split()
			if line.strip().startswith(";"):
				continue
			elif (len(words) == 2):
				top_data['moltype%s' % i] = {}
				top_data['moltype%s' % i]['resname'] = words[0]
				top_data['moltype%s' % i]['amount'] = words[1]
				i += 1
			else:
				break

	return top_data


def read_gro_file(grofile: str) -> dict:
	"""
	Extract the data from GROMACS (.gro) file.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions

	OUTPUT:
	gro_data - nested dictionary -> {info: {'header': str, 'natoms': int, 'boxDim': str},
	atom_number: {'molNumber': int, 'resName': str, 'atomName': str, 'atomsCoord': [float, float, float]}}
	"""

	# Create the nested dictionary to parse molecular structure data
	gro_data = {}

	# Define a atom number index
	i = 1

	with open(grofile, "r") as gro:
		line = gro.readline()

		# Parse header and number of atoms to info dictionary
		gro_data['info'] = {}
		gro_data['info']['header'] = line
		line = gro.readline()
		gro_data['info']['natoms'] = int(line.strip())
		line = gro.readline()

		# Parse the atomic data 
		while len(line.split()) >= 5:
			words = line.split()
			gro_data[i] = {}
			gro_data[i]['molNumber'] = int(line[0:5].strip())
			gro_data[i]['resName'] = line[5:10].strip()
			gro_data[i]['atomName'] = line[10:15].strip()
			gro_data[i]['atomsCoord'] = [float(line[20:28].strip()), float(line[28:36].strip()), float(line[36:44].strip())]
			i += 1
			line = gro.readline()

		# Parse the box dimensions
		gro_data['info']['boxDim'] = line

	return gro_data


def nMolecules(topfile: str) -> int:
	"""
	Reads the total number of molecules and solvent molecules from .top file.

	PARAMETRS:
	topfile - the .top file from GROMACS with number of molecules.

	OUTPUT:
	totalMolecules [type: int] - total number of molecules 
	solvMolecules [type: int] - number of solvent molecules 
	"""

	top_data = read_top_file(topfile)

	totalMolecules = 0
	for j in range(len(top_data)):
		totalMolecules += int(top_data['moltype%s' % j]['amount'])
		if top_data['moltype%s' % j]['resname'] == 'SOL':
			solvMolecules = int(top_data['moltype%s' % j]['amount'])

	return totalMolecules, solvMolecules


def output_writer(grofile: str, topfile: str, evapMolecules: list, cycle: int):
	"""
	Writes .gro and .top files after solvent removal.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules
	evapRate [type: float] - the rate of evaporation in percentage 
	cycle [type: int] - integer to track the evaporation loop 
	evapN [type: int] - n: n molecules to evaporate
						0: number of molecules is computed on-the-fly

	OUTPUT:
	A set of "cycleX*.gro" and "cycleX*.top" files with the molecular configuration and topology
	after each solvent removal step. 	
	"""

	# Create a nested dictionary with topology data
	top_data = read_top_file(topfile)

	# Create a nested dictionary with molecular structure data
	gro_data = read_gro_file(grofile)

	# Extract the number of molecules and solvent molecules
	totalMolecules, solvMolecules = nMolecules(topfile)

	# Determine the number of atoms in the solvent
	countSOL = 0
	for i in gro_data:
		for j, k in gro_data[i].items():
			if k == 'SOL':
				countSOL += 1

	initMolecules = solvMolecules

	# Write the updated "cycleX.gro" file after a cycle of solvent removal
	groout = open("cycle{0}.gro".format(cycle), "w")

	# Header and number of atoms
	header = gro_data['info']['header'].split("after")[0]

	groout.write("{0} after {1} cycles of solvent removal \n".format(header.strip(), cycle))
	groout.write(" " + str(gro_data['info']['natoms'] - int(countSOL/initMolecules)*len(evapMolecules)) + "\n")

	# Molecular and atomic data
	lastMol = 0
	countMol = 0
	countAtom = 0
	for i in gro_data:
		bool_ = True
		for j, k in gro_data[i].items():
			if i != 'info':
				if (j == 'molNumber') and (k in evapMolecules):
					if k != lastMol:
						countMol += 1
					lastMol = k
					countAtom += 1
					break
				elif bool_:
					groout.write('{:>5}{:<5}{:>5}{:>5}{:>8.3f}{:>8.3f}{:>8.3f}\n'.format(gro_data[i]['molNumber']-countMol, gro_data[i]['resName'], gro_data[i]['atomName'],
					 i-countAtom, gro_data[i]['atomsCoord'][0], gro_data[i]['atomsCoord'][1], gro_data[i]['atomsCoord'][2]))	# {:>6}{:<11}{:>7}{:>7}{:>7}{:>7}{:>11.4f}{:>11.4f}\n
					bool_ = False

	# Simulation box dimensions
	groout.write(gro_data['info']['boxDim'])

	# Change the number of solvent atoms in the topology file
	topout = open("cycle{0}.top".format(cycle), "w")

	with open(topfile) as top:
		line = top.readline()
		while "SOL" not in line:
			topout.write(line)
			line = top.readline()
		
		words = line.split()
		words[1] = str(int(words[1]) - len(evapMolecules))
		topout.write(words[0] + '\t' + words[1] + '\n')
		line = top.readline()

		while line:
			topout.write(line)
			line = top.readline()


def random_remover(grofile: str, topfile: str, evapRate: float, cycle: int, evapN: int):
	"""
	Randomly remove the solvent and generate new .gro and .top files.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules
	evapRate [type: float] - the rate of evaporation in percentage 
	cycle [type: int] - integer to track the evaporation loop 
	evapN [type: int] - n: n molecules to evaporate
						0: number of molecules is computed on-the-fly

	OUTPUT:
	A set of "cyclei_*.gro" and "cyclei_*.top" files with the molecular configuration and topology
	after each solvent removal step. 
	"""

	# Create a nested dictionary with topology data
	top_data = read_top_file(topfile)

	# Create a nested dictionary with molecular structure data
	gro_data = read_gro_file(grofile)

	# Extract the number of molecules and solvent molecules
	totalMolecules, solvMolecules = nMolecules(topfile)

	# Random number generator
	rng = np.random.default_rng()

	# Determine the solvent molecules to be removed
	if evapN == 0:
		evapMolecules = rng.choice(a=solvMolecules, size=int(round(evapRate*solvMolecules/100)), replace=False, shuffle=False).tolist()
	else:
		evapMolecules = rng.choice(a=solvMolecules, size=evapN, replace=False, shuffle=False).tolist()

	# Shift the random molecules to consider only solvent
	evapMolecules = [x+totalMolecules-solvMolecules+1 for x in evapMolecules]

	output_writer(grofile, topfile, evapMolecules, cycle)


def center_of_mass(atomic_dict: dict, resname: str) -> np.ndarray:
	"""
	Compute the center of mass from atoms of a given atomic dictionary.

	PARAMETERS:
	atomic_dict [type: dict] -> the dictionary with atomic data.
	resname [type: str] -> name of the residue to compute the center of mass

	OUTPUT:
	cm [type: np.ndarray] - array with the center of mass x, y and z coordinates.
	"""

	# Initialize the variables
	symbols = []
	_x = []
	_y = []
	_z = []

	# Loop over the dict
	for key, inner_dict in atomic_dict.items():

		# Check the type of input (list or string)
		if isinstance(resname, list):

			# Loop over the reference residues
			for name in resname:

				# Check for each reference residue
				if 'resName' in inner_dict and inner_dict['resName'] == name:

					# Get the atomic number and append it to Z
					_symbol = inner_dict['atomName'][:2]
					_symbol = _symbol if _symbol[1].islower() else _symbol[0]

					symbols.append(_symbol)

					# Get the coordinates
					_x.append(inner_dict['atomsCoord'][0])
					_y.append(inner_dict['atomsCoord'][1])
					_z.append(inner_dict['atomsCoord'][2])

		elif isinstance(resname, str):

			# Check for each reference residue
			if 'resName' in inner_dict and inner_dict['resName'] == resname:

				# Get the atomic number and append it to Z
				_symbol = inner_dict['atomName'][:2]
				_symbol = _symbol if _symbol[1].islower() else _symbol[0]

				symbols.append(_symbol)

				# Get the coordinates
				_x.append(inner_dict['atomsCoord'][0])
				_y.append(inner_dict['atomsCoord'][1])
				_z.append(inner_dict['atomsCoord'][2])

	# Get the masses
	masses = [Symbol_to_mass[i] for i in symbols]

	# Compute the center of mass
	coords = np.array([_x, _y, _z], dtype=np.float64)
	cm = np.average(coords, axis=1, weights=masses)

	return cm


def smallest_dist(ref_dict: dict, dict2: str) -> np.ndarray:
	"""
	Compute the smallest distance between atoms in dict2 with respect to atoms in ref_dict.

	PARAMETERS:
	ref_dict [type: dict] -> reference dictionary with atomic data.
	dict2 [type: str] -> dictionary with atomic data.

	OUTPUT:
	min_dist [type: float] - minimum distance between atoms in each dictionary.
	"""

	# Initialize the variables
	_x = []
	_y = []
	_z = []
	min_dist = 10**6

	# Get the atomic positions from ref_dict
	for _, inner_dict in ref_dict.items():
		_x.append(inner_dict['atomsCoord'][0])
		_y.append(inner_dict['atomsCoord'][1])
		_z.append(inner_dict['atomsCoord'][2])

	# Create a np.ndarray
	ref_coords = np.array([_x, _y, _z], dtype=np.float64).T

	# Get the smallest distance
	for _, inner_dict in dict2.items():

		# Loop over all atoms from reference
		for atom1 in ref_coords:

			# Create a np.ndarray with x, y and z coordinates of the current atom from dict2
			atom2 = np.array([inner_dict['atomsCoord'][0], inner_dict['atomsCoord'][1], inner_dict['atomsCoord'][2]], dtype=np.float64)
			
			# Compute the distance between atoms
			_dist = np.linalg.norm(atom1 - atom2)

			# Update the minimum distance
			if _dist < min_dist:
				min_dist = _dist

	return min_dist


def compute_dist(atomic_dict: dict, refname: str, rmname: str, cm: bool) -> np.ndarray:
	"""
	Compute the distance between centers of mass.

	PARAMETERS:
	atomic_dict [type: dict] -> the dictionary with atomic data.
	refname [type: str] -> name of the reference residue
	rmname [type: str] -> name of the residues to be removed
	cm [type: bool] -> If true, compute the distance between centers of mass.
					   If false, compute the distance between each atom.

	OUTPUT:
	dist [type: ndarray] - np.ndarray with the molNumbers and distances between center of masses.
	"""

	# Initialize the variables
	dist = np.array([]).reshape(0, 2)
	_molNumbers = np.array([])

	if cm == True:
		# Compute the reference center of mass
		cm_ref = center_of_mass(atomic_dict, refname)
	else:
		# Get a dict with the reference residue atoms
		if isinstance(refname, list):
			for name in refname:
				dict_ref = {key: in_dict for key, in_dict in atomic_dict.items() if 'resName' in in_dict and in_dict['resName'] == name}
		elif isinstance(refname, str):
			dict_ref = {key: in_dict for key, in_dict in atomic_dict.items() if 'resName' in in_dict and in_dict['resName'] == refname}

	# Loop over the dict
	for key, inner_dict in atomic_dict.items():

		# Check the type of input (list or string)
		if isinstance(rmname, list):

			# Loop over the reference residues
			for name in rmname:

				# Check for each reference residue
				if 'resName' in inner_dict and inner_dict['resName'] == name:

					# Create an array with molNumbers 
					if inner_dict['molNumber'] not in _molNumbers:

						_molNumbers = np.append(_molNumbers, inner_dict['molNumber'])

		elif isinstance(rmname, str):

			# Check for each reference residue
			if 'resName' in inner_dict and inner_dict['resName'] == rmname:

				# Create an array with molNumbers 
				if inner_dict['molNumber'] not in _molNumbers:

					_molNumbers = np.append(_molNumbers, inner_dict['molNumber'])

	# Loop over the candidates to be removed
	for num in _molNumbers:

		# Create a dictionary with the current molecule
		_dict = {key: in_dict for key, in_dict in atomic_dict.items() if 'molNumber' in in_dict and in_dict['molNumber'] == num}

		if cm == True:
			# Compute the current residue center of mass
			_cm = center_of_mass(_dict, rmname)

			# Compute the distance to the reference center of mass
			_dist = np.linalg.norm(_cm - cm_ref)

		else:
			# Compute the smallest distance between atoms from num and the reference residue
			_dist = smallest_dist(dict_ref, _dict)

		# Append it to the distance matrix
		dist = np.append(dist, [[num, _dist]], axis=0)

	return dist


def dist_rm_molecules(topfile: str, dist: np.ndarray, evapRate: float, evapN: int) -> list:
	"""
	Select the molecules to be removed using the distance criterion.

	PARAMETERS:
	topfile - the .top file from GROMACS with number of molecules
	dist [type: np.ndarray] - np.ndarray with molNumbers and distances
	evapRate [type: float] - the rate of evaporation in percentage 
	evapN [type: int] - n: n molecules to evaporate
						0: number of molecules is computed on-the-fly

	OUTPUT:
	rmMolecules [type: list] - list with molecules to be removed
	"""

	# Decrescent ordering o molecules with respect to distances
	ordered_dist = dist[np.argsort(dist[:, 1])[::-1]]

	if evapN == 0:

		# Extract the number of molecules to be removed
		_, rmMolecules = nMolecules(topfile)

		# Compute the amount of molecules to be removed
		_nMol = int(round(evapRate*rmMolecules/100, 0))
		
		# Clip the first _nMol molecules
		evapMolecules = ordered_dist[:_nMol, 0]

	else:
		# Clip the first evapN molecules
		evapMolecules = ordered_dist[:evapN, 0]

	return evapMolecules


def dist_remover(grofile: str, topfile: str, evapRate: float, cycle: int, evapN: int, refname: str, rmname: str, cm: bool):
	"""
	Randomly remove the solvent and generate new .gro and .top files.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules
	evapRate [type: float] - the rate of evaporation in percentage 
	cycle [type: int] - integer to track the evaporation loop 
	evapN [type: int] - n: n molecules to evaporate
						0: number of molecules is computed on-the-fly
	refname [type: str] - name of the reference residue
	rmname [type: str] - name of the residue to be removed
	cm [type: bool] -> If true, compute the distance between centers of mass.
					   If false, compute the distance between each atom.

	OUTPUT:
	A set of "cyclei_*.gro" and "cyclei_*.top" files with the molecular configuration and topology
	after each solvent removal step. 
	"""

	# Create a nested dictionary with topology data
	top_data = read_top_file(topfile)

	# Create a nested dictionary with molecular structure data
	gro_data = read_gro_file(grofile)

	# Extract the number of molecules and solvent molecules
	totalMolecules, solvMolecules = nMolecules(topfile)

	# Get a vector with the distance of each center of mass to the reference center of mass
	dist_vect = compute_dist(gro_data, refname, rmname, cm)
	print(dist_vect)

	# Create evapMolecules list with molNumbers to be removed
	evapMolecules = dist_rm_molecules(topfile, dist_vect, evapRate, evapN)

	# Write the .gro and .top files
	output_writer(grofile, topfile, evapMolecules, cycle)


def setup_lince2(cycle: int):
	"""
	Set the environment and run GROMACS simulations at the HPC - Lince2 cluster.

	PARAMETERS:
	cycle [type: int] - integer to track the evaporation loop

	OUTPUT:
	None
	"""

	# # Run the preprocessor to generate GROMACS binaries
	# sp.run("gmx grompp -f npt.mdp -c cycle{0}.gro -p cycle{0}.top -o cycle{0}.tpr -po mdout{0}.mdp".format(cycle), shell=True)

	# # Run the NPT simulation
	# sp.run("gmx mdrun -s cycle{0}.tpr -deffnm CYCLE{0}-NPT -pin on -ntmpi 2 -ntomp 8 -maxh 0.05".format(cycle), shell=True)

	# # Create a directory for the current cycle
	# sp.run("mkdir cycle{0}".format(cycle), shell=True)

	# # Copy the input files to the current cycle directory
	# # sp.run("cp npt.mdp cycle{0}.gro cycle{0}.top cycle{0}/".format(cycle), shell=True)
	# sp.run("cp npt.mdp cycle{0}/".format(cycle), shell=True)
	# sp.run("mv mdout{0}.mdp cycle{0}.* CYCLE{0}-NPT.* cycle{0}/".format(cycle), shell=True)

	# Create a directory for the current cycle
	sp.run("mkdir cycle{0}".format(cycle), shell=True)

	# Copy the input files to the current cycle directory
	# sp.run("cp npt.mdp cycle{0}.gro cycle{0}.top cycle{0}/".format(cycle), shell=True)
	sp.run("cp npt.mdp cycle{0}/".format(cycle), shell=True)
	sp.run("mv cycle{0}.gro cycle{0}.top cycle{0}/".format(cycle), shell=True)

	# Go to the current cycle directory
	# sp.run("cd cycle{0}".format(cycle), shell=True)

	# Run the preprocessor to generate GROMACS binaries
	# sp.run("gmx grompp -f npt.mdp -c cycle{0}.gro -p cycle{0}.top -o cycle{0}.tpr".format(cycle), shell=True)

	# Run the NPT simulation
	# sp.run("gmx mdrun -s cycle{0}.tpr -deffnm CYCLE{0}-NPT -pin on -ntmpi 2 -ntomp 8 -maxh 0.05".format(cycle), shell=True)

	# Go back to the script directory
	# sp.run("cd ../", shell=True)

	# TESTES
	sp.run("cp cycle{0}/cycle{0}.gro cycle{0}/CYCLE{0}-NPT.gro".format(cycle), shell=True)


def setup_lovelace_1gpu(cycle: int):
	"""
	Set the environment and run GROMACS simulations in 'umagpu' queue at the CENAPAD - Lovelace cluster.

	PARAMETERS:
	cycle [type: int] - integer to track the evaporation loop

	OUTPUT:
	None
	"""

	# Run the preprocessor to generate GROMACS binaries
	sp.run("$nv_gromacs gmx grompp -f npt.mdp -c cycle{0}.gro -p cycle{0}.top -o cycle{0}.tpr -po mdout{0}.mdp -maxwarn 2".format(cycle), shell=True)
	
	# Run the NPT simulation
	sp.run("OMP_NUM_THREADS=4 $nv_gromacs gmx mdrun -s cycle{0}.tpr -deffnm CYCLE{0}-NPT -c CYCLE{0}-NPT.gro -ntmpi 4 -nb gpu -pin on -v -ntomp 4 -notunepme -resethway".format(cycle), shell=True)

	# Create a directory for the current cycle
	sp.run("mkdir cycle{0}".format(cycle), shell=True)

	# Copy the input files to the current cycle directory
	sp.run("cp npt.mdp cycle{0}/".format(cycle), shell=True)
	sp.run("mv mdout{0}.mdp cycle{0}.* CYCLE{0}-NPT* cycle{0}/".format(cycle), shell=True)


def solvent_evaporation(grofile: str, topfile: str, evapRate: float, evapTotal: float, cluster: str, dynamic: bool, thermo: bool, refname: str, rmname: str):
	"""
	Perform a loop of solvent removal and GROMACS NPT simulation.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules
	evapRate [type: float] - the rate of evaporation in percentage 
	evapTotal [type: float] - the total evaporation in percentage 
	cluster [type: str] - specify which cluster the calculation is running into ('lince2' or 'lovelace')
	dynamic [type: bool] - TRUE: number of molecules to be removed is computed on-the-fly
						   FALSE: number of molecules to be removed is fixed, based on the initial amount of solvent molecules
	thermo [type: bool]	- TRUE: thermodynamical solvent removal is performed
						  FALSE: random solvent removal is performed
	refname [type: str] - name of the reference residue
	rmname [type: str] - name of the residue to be removed

	OUTPUT:
	A set of "cyclei*.gro" files for each solvent removal step,
	with final configuration after complete solvent removal named "evaporated_*.gro".
	"""

	if dynamic and thermo:
		print("Thermodynamical solvent removal is not compatible with random removal on-the-fly calculations. Please choose one.")
		exit()

	# Nested dictionary with topology data
	top_data = read_top_file(topfile)

	# Nested dictionary with molecular structure data
	gro_data = read_gro_file(grofile)

	# Number of molecules and solvent molecules
	totalMolecules, solvMolecules = nMolecules(topfile)

	# Number of solvent molecules to evaporate at the initial cycle
	evapInit = round(evapRate*solvMolecules/100)

	# First solvent removal
	print("Working on cycle 1.")
	if thermo:
		thermo_remover(grofile, topfile, evapRate, 1, cluster)
	else:
		# random_remover(grofile, topfile, evapRate, 1, 0)
		dist_remover(grofile, topfile, evapRate, 1, 0, refname, rmname)

	# First NPT simulation
	if cluster == 'lince2':
		setup_lince2(1)
	elif cluster == 'lovelace':
		setup_lovelace_1gpu(1)

	if dynamic:
		#########################################################################
		### Cycle using evapRate % over the current total number of molecules ###
		#########################################################################

		i = 1
		while (solvMolecules >= evapTotal*solvMolecules/100) and (solvMolecules > 1):
			print("Working on cycle %s." % str(i+1))
			top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
			gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
			totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))

			if (evapRate*totalMolecules/100 >= 1):
				# random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 0)
				dist_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 0, refname, rmname)
			else:
				# random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 1)
				dist_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 1, refname, rmname)
			
			if cluster == 'lince2':
				setup_lince2(i+1)
			elif cluster == 'lovelace': 
				setup_lovelace_1gpu(i+1)
			i += 1

	if thermo:
		#######################################################
		### Cycle using thermodynamical solvent evaporation ###
		#######################################################

		i = 1
		while solvMolecules > 0:
			sp.run("mv mdout{0}.mdp *{0}_2z.* CYCLE{0}-NVT* cycle{0}/".format(i), shell=True)

			print("Working on cycle %s." % str(i+1))
			top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
			gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
			totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))

			thermo_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, cluster)

			if cluster == 'lince2':
				setup_lince2(i+1)
			elif cluster == 'lovelace':
				setup_lovelace_1gpu(i+1)
			i += 1

	else:
		#########################################################################
		### Cycle using evapRate % over the initial total number of molecules ###
		#########################################################################

		i = 1
		while solvMolecules > 2*evapInit:
			print("Working on cycle %s." % str(i+1))
			top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
			gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
			totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))

			# random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, evapInit)
			dist_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, evapInit, refname, rmname)
			
			if cluster == 'lince2':
				setup_lince2(i+1)
			elif cluster == 'lovelace':
				setup_lovelace_1gpu(i+1)
			i += 1

		# Remove the remaining solvent molecules
		print("Working on cycle %s." % str(i+1))
		top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
		gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
		totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))
		# random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, solvMolecules)
		dist_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, solvMolecules, refname, rmname)
		
		if cluster == 'lince2':
			setup_lince2(i+1)
		elif cluster == 'lovelace': 
			setup_lovelace_1gpu(i+1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Recieves a Gaussian output file and generate a .csv file.")
	parser.add_argument("grofile", help="the .gro input file.")
	parser.add_argument("topfile", help="the .top topology file.")
	parser.add_argument("evapTotal", type=float, help="the total percentage of evaporation.")
	parser.add_argument("evapRate", type=float, help="the evaporation rate (in percentage).")
	parser.add_argument("cluster", type=str, help="the cluster where calculations are going to be performed (lince2 or lovelace)")
	parser.add_argument("--refname", "-rf", nargs="+", type=str, help="name of the reference residue.", default='UNL')
	parser.add_argument("--rmname", "-rm", nargs="+", type=str, help="name of residue to be removed.", default='SOL')
	parser.add_argument("--dynamic", "-d", help="number of molecules to evaporate is computed on-the-fly.", action="store_true")
	parser.add_argument("--thermo", "-t", help="thermodynamical solvent removal is performed \
						(random removal is default).", action="store_true")
	parser.add_argument("--center-of-mass", "-cm", help="If True, uses center of mass distances.", action="store_true", default=False)

	args = parser.parse_args()

	# solvent_evaporation(args.grofile, args.topfile, args.evapRate, args.evapTotal, args.cluster, args.dynamic, args.thermo, args.refname, args.rmname)

	dist_remover(args.grofile, args.topfile, args.evapRate, 1, 0, args.refname, args.rmname, args.center_of_mass)
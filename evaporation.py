#!/usr/bin/env python3

"""
Script to perform evaporation of solvent using GROMACS.

[X] Remove randomly a given percentage of solvent and create a new .gro file.
[X] Update the number of solvent molecules in the topology (.top) file.
[X] Run a NPT simulation.
[X] Make a loop for complete solvent elimination.

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
		while len(line.split()) == 6:
			words = line.split()
			gro_data[i] = {}
			gro_data[i]['molNumber'] = int(line[0:5].strip())
			gro_data[i]['resName'] = line[5:10].strip()
			gro_data[i]['atomName'] = line[10:15].strip()
			gro_data[i]['atomsCoord'] = [float(words[3]), float(words[4]), float(words[5])]
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
	1) A set of "cyclei_*.gro" and "cyclei_*.top" files with the molecular configuration and topology
	after each solvent removal step. 
	2) A "evaporated_*.gro" file with final configuration after complete solvent removal.
	"""

	# Create a nested dictionary with topology data
	top_data = read_top_file(topfile)

	# Create a nested dictionary with molecular structure data
	gro_data = read_gro_file(grofile)

	# Extract the number of molecules and solvent molecules
	totalMolecules, solvMolecules = nMolecules(topfile)

	# Determine the number of atoms in the solvent and parse it to the info dictionary
	countSOL = 0
	for i in gro_data:
		for j, k in gro_data[i].items():
			if k == 'SOL':
				countSOL += 1

	initMolecules = solvMolecules

	# Random number generator
	rng = np.random.default_rng()

	# Determine the solvent molecules to be removed
	if evapN == 0:
		evapMolecules = rng.choice(a=solvMolecules, size=int(round(evapRate*solvMolecules/100)), replace=False, shuffle=False).tolist()
	else:
		evapMolecules = rng.choice(a=solvMolecules, size=evapN, replace=False, shuffle=False).tolist()

	# Shift the random molecules to consider only solvent
	evapMolecules = [x+totalMolecules-solvMolecules+1 for x in evapMolecules]

	# Write the updated .gro file after a cycle of solvent removal
	# groout = open("cycle{0}_{1}".format(str(cycle), os.path.basename(grofile).split("_")[-1]), "w")
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
					 i-countAtom, gro_data[i]['atomsCoord'][0], gro_data[i]['atomsCoord'][1], gro_data[i]['atomsCoord'][2]))
					bool_ = False

	# Simulation box dimensions
	groout.write(gro_data['info']['boxDim'])

	# Change the number of solvent atoms in the topology file
	# topout = open("cycle{0}_{1}".format(str(cycle), os.path.basename(topfile).split("_")[-1]), "w")
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

def setup_lince2(cycle: int):
	"""
	Create directories and files for running GROMACS simulations.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules.
	cycle [type: int] - integer to track the evaporation loop

	OUTPUT:
	runEvaporation.sh - submission script file for lince2 cluster
	"""

	# Create a directory for the current cycle
	sp.run("mkdir cycle{0}".format(cycle), shell=True)

	# Copy the input files to the current cycle directory
	sp.run("cp npt.mdp cycle{0}/".format(cycle), shell=True)
	sp.run("mv cycle{0}.gro cycle{0}.top cycle{0}/".format(cycle), shell=True)

	# Go to the current cycle directory
	sp.run("cd cycle{0}".format(cycle), shell=True)

	# Run the preprocessor to generate GROMACS binaries
	sp.run("gmx grompp -f npt.mdp -c cycle{0}.gro -p cycle{0}.top -o cycle{0}.tpr".format(cycle), shell=True)

	# Run the NPT simulation
	sp.run("gmx mdrun -s cycle{0}.tpr -deffnm CYCLE{0}-NPT -pin on -ntmpi 2 -ntomp 8 -maxh 0.05".format(cycle), shell=True)

	# Go back to the script directory
	sp.run("cd ../", shell=True)

def solvent_evaporation(grofile: str, topfile: str, evapRate: float, evapTotal: float, dynamic: bool):
	"""
	Perform a loop of solvent removal and GROMACS NPT simulation.

	PARAMETERS:
	grofile - the .gro file from GROMACS with atomic positions
	topfile - the .top file from GROMACS with number of molecules
	evapRate [type: float] - the rate of evaporation in percentage 
	evapTotal [type: float] - the total evaporation in percentage 
	dynamic [type: bool] - TRUE: number of molecules to be removed is computed on-the-fly
						   FALSE: number of molecules to be removed is fixed, based on the initial amount of solvent molecules

	OUTPUT:
	A set of "cyclei*.gro" files for each solvent removal step,
	with final configuration after complete solvent removal named "evaporated_*.gro".
	"""

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
	random_remover(grofile, topfile, evapRate, 1, 0)

	# First NPT simulation
	setup_lince2(1)

	if dynamic:
		#########################################################################
		### Cycle using evapRate % over the current total number of molecules ###
		#########################################################################

		i = 1
		while (solvMolecules >= evapTotal*solvMolecules/100) and (solvMolecules > 1):
			print("Working on cycle %s." % str(i+1))
			# if i == 1:
			# 	top_data = read_top_file("cycle{0}_{1}".format(str(i), topfile))
			# 	gro_data = read_gro_file("cycle{0}_{1}".format(str(i), grofile))
			# 	totalMolecules, solvMolecules = nMolecules("cycle{0}_{1}".format(str(i), topfile))
			# 	if (evapRate*totalMolecules/100 >= 1):
			# 		random_remover("cycle{0}_{1}".format(str(i), grofile), "cycle{0}_{1}".format(str(i), topfile), evapRate, i+1, 0)
			# 	else:
			# 		random_remover("cycle{0}_{1}".format(str(i), grofile), "cycle{0}_{1}".format(str(i), topfile), evapRate, i+1, 1)
			# else:
			top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
			gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
			totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))
			if (evapRate*totalMolecules/100 >= 1):
				random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 0)
			else:
				random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, 1)
			setup_lince2(i+1)
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
			random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, evapInit)
			setup_lince2(i+1)
			i += 1

		# Remove the remaining solvent molecules
		print("Working on cycle %s." % str(i+1))
		top_data = read_top_file("cycle{0}/cycle{0}.top".format(i))
		gro_data = read_gro_file("cycle{0}/CYCLE{0}-NPT.gro".format(i))
		totalMolecules, solvMolecules = nMolecules("cycle{0}/cycle{0}.top".format(i))
		random_remover("cycle{0}/CYCLE{0}-NPT.gro".format(i), "cycle{0}/cycle{0}.top".format(i), evapRate, i+1, solvMolecules)
		setup_lince2(i+1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Recieves a Gaussian output file and generate a .csv file.")
	parser.add_argument("grofile", help="the .gro input file.")
	parser.add_argument("topfile", help="the .top topology file.")
	parser.add_argument("evapTotal", type=float, help="the total percentage of evaporation.")
	parser.add_argument("evapRate", type=float, help="the evaporation rate (in percentage).")
	parser.add_argument("--dynamic", "-d", help="if number of molecules to evaporate is computed on-the-fly.", action="store_true")

	args = parser.parse_args()

	solvent_evaporation(args.grofile, args.topfile, args.evapRate, args.evapTotal, args.dynamic)

	# print(type(args.grofile), type(args.topfile), type(args.evapRate), type(args.evapTotal), type(args.dynamic))

	# random_remover(args.grofile, args.topfile, args.evapRate)
	# read_top_file(args.topfile)
	# read_gro_file(args.grofile)
	# setup_lince2(args.grofile, args.topfile, 1)

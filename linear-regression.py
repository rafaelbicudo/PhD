#!/usr/bin/env python3 

"""
Script do the following:

[X] 1) Extract QM energy from torsional scan via Gaussian09/16 and write a "qm_scan.csv" file 
	-> 2 columns with angles and energies.
[X] 2) Determine all dihedral angles that change during the scan and write a "dihedrals.csv" file 
	-> columns with changing torsionals for each configuration.
[X] 3) Interpolate the QM data using a cubic polynomial and perform a linear regression to determine the 4 coefficients in 
the Fourier version of Ryckaert-Bellemans torsional potentials (F_1, F_2, F_3 and F_4). -> Actually using the 6 coefficients
that yielded better results.

Author: Rafael Bicudo Ribeiro (@rafaelbicudo) and Thiago Duarte (@thiagodsd)
DATE: DEZ/2022
"""

import argparse
import numpy as np
import os

from plot_en_angle_gaussian_scan import parse_en_log_gaussian
from plot_eff_tors import get_phi
from parse_gaussian_charges import find_natoms

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model  import LinearRegression
from scipy.interpolate import CubicSpline

def find_bonded_atoms(topfile, a1, a2, a3, a4):
	"""
	Return a list of all candidates to change during the rotation of a1-a2-a3-a4 angle.
	
	PARAMETERS:
	topfile - topology file
	a1 - first atom defining the dihedral angle
	a2 - second atom defining the dihedral angle
	a3 - third atom defining the dihedral angle
	a4 - fourth atom defining the dihedral angle

	OUTPUT:
	Angles (list)
	"""

	dih_atoms = [a1, a2, a3, a4]
	dih_atoms = [int(i) for i in dih_atoms]

	_atoms = dih_atoms.copy()

	with open(topfile, "r") as f:
		line = f.readline()
		while "[ bonds ]" not in line:
			line = f.readline()

		line = f.readline()
		while (len(line.split()) != 0) and not (line.strip().startswith("[")):
			words = line.split()

			if (int(words[0]) in dih_atoms) and (int(words[1]) not in dih_atoms):
				_atoms.append(int(words[1]))

			elif (int(words[1]) in dih_atoms) and (int(words[0]) not in dih_atoms):
				_atoms.append(int(words[0]))
			line = f.readline()

	return _atoms

def find_dihedrals(topfile, a1, a2, a3, a4):
	"""
	Find which torsionals are candidates for changing during the scan.

	PARAMETERS:
	topfile - topology file
	a1 - first atom defining the dihedral angle
	a2 - second atom defining the dihedral angle
	a3 - third atom defining the dihedral angle
	a4 - fourth atom defining the dihedral angle

	OUTPUT: 
	Dihedral angles (list of lists with 4 integers each) and labels (list of strings)
	"""

	candidates = find_bonded_atoms(topfile, a1, a2, a3, a4)

	dih_atoms = [a1, a2, a3, a4]
	dih_atoms = [int(i) for i in dih_atoms]

	tors = []
	tors_label = []

	with open(topfile, "r") as f:
		line = f.readline()
		while "[ dihedrals ]" not in line:
			line = f.readline()

		while True:
			line = f.readline()
			if line.strip().startswith(";"):
				continue
			if (len(line.split()) == 11) and (int(line.split()[4]) != 2) and (int(line.split()[4]) != 4):
				break
			if not line:
				print("There are no proper dihedrals.")
				break

		while (len(line.split()) != 0) and not (line.strip().startswith("[")):
			words = line.split()
			_atoms = [words[0], words[1], words[2], words[3]]
			_atoms = [int(i) for i in _atoms]
			if all(elem in candidates for elem in _atoms):
				tors_label.append("%s-%s-%s-%s" % (words[0], words[1], words[2], words[3]))
				tors.append([int(words[0]), int(words[1]), int(words[2]), int(words[3])])
			line = f.readline()

	return tors, tors_label

def write_torsional_changes(xyzrotationsfile, topfile, a1, a2, a3, a4, npoints):
	"""
	Read the .xyz file and find torsional angle changes.

	PARAMETERS:
	xyzrotationsfile - configurations file
	topfile - topology file
	a1 - first atom defining the dihedral angle
	a2 - second atom defining the dihedral angle
	a3 - third atom defining the dihedral angle
	a4 - fourth atom defining the dihedral angle

	OUTPUT:
	The "dihedrals.csv" file.
	"""

	atomsCoord = {}
	i = 0

	# Get the number of atoms in the molecule
	natoms = find_natoms(topfile)

	# Get the torsional angles and labels
	tors, tors_label = find_dihedrals(topfile, a1, a2, a3, a4)

	# Create the data frame with proper size
	df = pd.DataFrame(np.zeros((npoints, len(tors))), index=range(1, npoints+1), columns=tors_label)

	with open(xyzrotationsfile) as xyz_f:
		line = xyz_f.readline()
		words = line.split()

		for conf in range(npoints):
			while len(words) != 4:
				line = xyz_f.readline()
				words = line.split()

			# Parse the atomic coordinates to a dictionary (adapted from "parse_txt" function from DICEtools)
			anum = 1
			for i in range(natoms):
				atomsCoord[anum] = [float(x) for x in line.split()[1:4]]
				anum += 1
				line = xyz_f.readline()
				words = line.split()

			# Add the torsional angles to the dataframe
			for _list in tors:
				df.iat[conf, tors.index(_list)] = get_phi(atomsCoord[_list[0]], atomsCoord[_list[1]], atomsCoord[_list[2]], atomsCoord[_list[3]])
			conf += 1

		for label in tors_label:
			column = round(df[label], 4).to_numpy()
			if (column == column[0]).all():
				del df[label]

	# print(df.columns[0], type(df.columns[0])) 
	df = df.sort_values(by=df.columns[0], ascending=True)

	df.to_csv('dihedrals.csv')

def write_dih_csv(gaussianlogfile):
	"""
	Read the gaussian output file and write the .csv file with dihedrals and energies.

	PARAMETERS:
	gaussianlogfile - Output file from Gaussian09/16

	OUTPUT:
	A "qm_scan.csv" file.
	"""

	# Read the dihedrals (in degrees) and energies (in kcal/mol)
	died, en = parse_en_log_gaussian(gaussianlogfile)

	# Rescale the energy to find the "QM" torsional energy
	en = [x-min(en) for x in en]

	fout = open('qm_scan.csv', 'w')

	fout.write(',Dihedral (rad),Energy - E_min (kcal/mol) \n')
	for i in range(len(died)):
			fout.write(str(i+1) + ',' + str(died[i]*np.pi/180) + ',' + str(en[i]) + '\n')

def write_itp_file(topfile, lr_data):
	"""
	Write the linear regression coefficients in the topology file.

	PARAMETERS:
	topfile - topology file
	lr_data - nested dictionary with atoms and coeficients for each torsional angle

	OUTPUT:
	A "LR_*.itp" file.
	"""

	fout = open("LR_" + os.path.basename(topfile), "w")
	fout.write("; ########################################################################### \n")
	fout.write("; ### LINEAR REGRESSION WAS PERFORMED TO DETERMINE TORSIONAL COEFFICIENTS ### \n")
	fout.write("; ########################################################################### \n")

	with open(topfile, "r") as f:
		line = f.readline()

		while "[ dihedrals ]" not in line:
			fout.write(line)
			line = f.readline()

		while "[ pairs ]" not in line:
			fout.write(line)
			line = f.readline()

			# Change the data from kcal/mol to kJ/mol and parse it to the topology file.
			if len(line.split()) == 11:
				words = line.split()
				if words[4] == '3':
					_tors = [words[0], words[1], words[2], words[3]]
					for k in range(len(lr_data)):
						if all(elem in lr_data[k]["atoms"] for elem in _tors):
							line = line.replace(words[5], str(round(lr_data[k]["constants"][0]*4.184, 3)), 1)
							line = line.replace(words[6], str(round(lr_data[k]["constants"][1]*4.184, 3)), 1)
							line = line.replace(words[7], str(round(lr_data[k]["constants"][2]*4.184, 3)), 1)
							line = line.replace(words[8], str(round(lr_data[k]["constants"][3]*4.184, 3)), 1)
							line = line.replace(words[9], str(round(lr_data[k]["constants"][4]*4.184, 3)), 1)
							line = line.replace(words[10], str(round(lr_data[k]["constants"][5]*4.184, 3)), 1)
				else:
					print("Not a Ryckaert-Bellemans dihedral.")

	# print(lr_data)

def linear_regression(topfile):
	"""
	Perform the linear regression, generate a plot with classical and "quantum" torsional energies
	and change the C_i coefficients in the topology file.

	PARAMETERS:
	None (reads *.csv files)

	Output:
	The "linear_regression.png" and "LR_*.itp" files.
	"""

	ans = pd.read_csv("qm_scan.csv", index_col=0)
	data = pd.read_csv("dihedrals.csv", index_col=0)

	# Cubic polynomial interpolation
	interpolarizer = CubicSpline(ans.iloc[:,0], ans.iloc[:,1])

	# Add the interpolation in the dataFrame data
	data["ans"] = data.apply( lambda x: interpolarizer(x[data.columns[0]]), axis=1)

	# Create a new data frame with ans values
	df = data[["ans"]].copy().rename(columns={"ans":"y"})

	# # Compute each Ryckaert-Bellemans torsional element
	# phi_1 = 0
	# phi_2 = 0
	# phi_3 = 0
	# phi_4 = 0

	# for c in data.drop(columns=["ans"]).columns:
	# 	df[f"{c}_V1"] = 0.5 * (1 + np.cos(1 * data[c] - phi_1))
	# 	df[f"{c}_V2"] = 0.5 * (1 - np.cos(2 * data[c] - phi_2))
	# 	df[f"{c}_V3"] = 0.5 * (1 + np.cos(3 * data[c] - phi_3))
	# 	df[f"{c}_V4"] = 0.5 * (1 - np.cos(4 * data[c] - phi_4))

	for c in data.drop(columns=["ans"]).columns:
		df[f"{c}_C0"] = np.cos(data[c])**0
		df[f"{c}_C1"] = np.cos(data[c])**1
		df[f"{c}_C2"] = np.cos(data[c])**2
		df[f"{c}_C3"] = np.cos(data[c])**3
		df[f"{c}_C4"] = np.cos(data[c])**4
		df[f"{c}_C5"] = np.cos(data[c])**5

	X = df.drop(columns=["y"])
	y = df["y"]

	# Apply the linear regression model
	reg = LinearRegression(fit_intercept=False)
	reg.fit(X,y)

	# Create a nested dictionary with the linear regression data
	lr_data = {}	# {i: {'tors': str, 'atoms': list, 'constants': list}}

	for i in range(int(len(reg.coef_)/6)):
		lr_data[i] = {}
		lr_data[i]["tors"] = X.columns.values.tolist()[6*i][:len(X.columns.values.tolist()[6*i])-3]
		lr_data[i]["atoms"] = []
		lr_data[i]["atoms"].extend([elem for elem in lr_data[i]["tors"].split('-')])
		lr_data[i]["constants"] = []
		for j in range(6):
			lr_data[i]["constants"].append(reg.coef_[6*i+j])

	# Plot the results
	x_plot = data[data.columns[0]].copy()
	y_plot = np.dot(X.values, reg.coef_)

	fig, ((ax1)) = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

	ax1.plot(ans[ans.columns[0]], ans[ans.columns[1]], "x--", c="r", label="gaussian_torsional")
	ax1.plot(x_plot, y_plot, "x--", c="b", label="regressao_linear")

	plt.legend()
	plt.tight_layout()
	plt.savefig("linear_regression.png")

	# Write the linear regression coefficients in the topology file
	write_itp_file(topfile, lr_data)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Recieves a Gaussian output file and generate a .csv file.")
	parser.add_argument("gaussianlogfile", help="the gaussian log file.")
	parser.add_argument("xyzrotationsfile", help="file with all torsional scan configurations.")
	parser.add_argument("topfile", help="the topology file.")
	parser.add_argument("a1", type=int, help="first atom defining the reference dihedral.")
	parser.add_argument("a2", type=int, help="second atom defining the reference dihedral.")
	parser.add_argument("a3", type=int, help="third atom defining the reference dihedral.")
	parser.add_argument("a4", type=int, help="fourth atom defining the reference dihedral.")
	parser.add_argument("npoints", type=int, help="number of configurations during the scan.")

	args = parser.parse_args()

	write_dih_csv(args.gaussianlogfile)

	write_torsional_changes(args.xyzrotationsfile, args.topfile, args.a1, args.a2, args.a3, args.a4, args.npoints)

	linear_regression(args.topfile)


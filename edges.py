#!/usr/bin/env python3

"""
Script to extract and plot core orbital energies.

AUTHOR: Rafael Bicudo Ribeiro (@rafaelbicudo) and Lucas Cornetta (@lmcornetta)
DATE: JUL/2023
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def nElectrons(logfile: str) -> int:
	"""
	Extract the number of alpha and beta electrons from Gaussian output files.

	PARAMETERS:
	logfile [type: str] - Gaussian log file.

	OUTPUT:
	nAlpha [type: int] - Number of electrons with alpha spin
	nBeta [type: int] - Number of electrons with beta spin
	"""

	nAlpha = 0
	nBeta = 0

	# Open and read the log file
	f = open(logfile, "r")
	lines = f.readlines()

	# Loop over lines and get the number of alpha/beta electrons
	for line in lines:
		if 'alpha electrons' in line:
			nAlpha = int(line.split()[0])
			nBeta = int(line.split()[3])

	return nAlpha, nBeta

def occOrbEnergies(logfile: str):
	"""
	Extract the occupied orbital energies from Gaussian output files.

	PARAMETERS:
	logfile [type: str] - Gaussian log file.

	OUTPUT:
	mo_list [type: np.array] - Array with molecular orbital energies.
	"""

	nAlpha, _ = nElectrons(logfile)

	mo_list = np.empty((0, nAlpha), dtype=np.float64)
	mo_conf = np.array([])

	f = open(logfile, "r")
	line = f.readline()

	# Loop over all lines
	while line:
		while ' The electronic state is' not in line:
			line = f.readline()
			if not line:
				break

		line = f.readline()
		mo_conf = np.array([])

		# Append orbital energies to mo_conf 
		while 'occ.' in line:
			mo_conf = np.append(mo_conf, np.array(line.split()[4:], dtype=np.float64))
			line = f.readline()

		if mo_conf.size != 0:
			mo_list = np.vstack((mo_list, mo_conf))

		line = f.readline()

	f.close()

	# Change to eV
	mo_list = 27.2114*mo_list

	return mo_list

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Estimate n-edge from orbital energies.")
	parser.add_argument("logfile", help="the gaussian log file.")
	parser.add_argument("--C1s", help="plot candidates for carbon K-edge", action='store_true', default=False)
	parser.add_argument("--O1s", help="plot candidates for oxygen K-edge", action='store_true', default=False)
	parser.add_argument("--N1s", help="plot candidates for nitrogen K-edge", action='store_true', default=False)
	parser.add_argument("--S1s", help="plot candidates for sulfur K-edge", action='store_true', default=False)
	parser.add_argument("--S2s", help=r"plot candidates for sulfur L$_1$-edge", action='store_true', default=False)
	parser.add_argument("--S2p", help=r"plot candidates for sulfur L$_{2,3}$-edge", action='store_true', default=False)

	args = parser.parse_args()

	mo_list = occOrbEnergies(args.logfile)

	# print(mo_list)

	angle = np.linspace(0, 360, 35)

	fig, ax = plt.subplots(4, 1, sharex=True)

	ax[0].set_title("oxygen K-edge", fontsize=16)
	for i in range(8, 10):
	    ax[0].plot(angle, np.transpose(mo_list)[i])
	ax[0].set_ylabel("bind. ene. (eV)", fontsize=14)

	ax[1].set_title("carbon K-edge", fontsize=16)
	for i in range(10, 78):
		ax[1].plot(angle, np.transpose(mo_list)[i])
	ax[1].set_ylabel("bind. ene. (eV)", fontsize=14)

	ax[2].set_title(r"sulfur $L_1$-edge", fontsize=16)
	for i in range(78, 86):
		ax[2].plot(angle, np.transpose(mo_list)[i])
	ax[2].set_ylabel("bind. ene. (eV)", fontsize=14)

	ax[3].set_title(r"sulfur $L_{2,3}$-edge", fontsize=16)
	for i in range(86, 110):
		ax[3].plot(angle, np.transpose(mo_list)[i])
	ax[3].set_ylabel("bind. ene. (eV)", fontsize=14)
	ax[3].set_xlabel("torsional angle (deg)", fontsize=14)

	fig.tight_layout()
	plt.show()

	fig.savefig("edges.png", bbox_inches='tight', format='png', dpi=300)



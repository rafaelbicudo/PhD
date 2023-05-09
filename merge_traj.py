#!/usr/bin/env python3

"""
Script to merge cycle*-traj.gro with trajectory data from evaporation cycles from evaporation.py script.

AUTHOR: Rafael Bicudo Ribeiro
DATE: 05/2023
"""

import argparse
import re

def get_t_and_step(string: str):
	"""
	Return column values corresponding to t and step.

	PARAMETERS:
	grotrajfile [type: str] - the .gro file with trajectory data

	OUTPUT:
	t [type: float] - the current time of simulation.
	step [type: int] - the current step of the simulation.
	"""

	words = string.split()
	for i in range(len(words)):
		if words[i] == "t=":
			t = float(words[i + 1])
			step = int(words[i + 3])

	return t, step

def find_dt_and_dstep(grotrajfile: str):
	"""
	Find dt and dstep at each trajectory .gro file.

	PARAMETERS:
	grotrajfile [type: str] - the .gro file with trajectory data

	OUTPUT:
	dt [type: float] - the time step between two consecutive data records from GROMACS.
	dstep [type: int] - the amout of simulation steps between two consecutive data records from GROMACS.
	"""

	i = 0

	with open(grotrajfile, "r") as f:
		line = f.readline()

		while line:
			if "t= " in line:
				dt, dstep = get_t_and_step(line)
				i += 1

			line = f.readline()

			if i > 1:
				f.close()
				return dt, dstep

def merge_gro_files(ncycles: int):
	"""
	Merge the cycle*-traj.gro files.

	PARAMETERS:
	ncycles [type: int] - the number of evaporation cycles

	OUTPUT:
	A file named "merged-traj.gro" with all trajectory frames combined.
	"""

	final_t = 0.0
	final_step = 0

	merged = open('merged-traj.gro', "w")

	# Loop over the first cycle
	c1 = open('cycle1-traj.gro', "r")

	line = c1.readline()

	while line:
		if "t= " in line:
			words = line.split()
			for k in range(len(words)):
				if words[k] == "t=":
					t = words[k + 1]
					step = words[k + 3]

		merged.write(line)
		line = c1.readline()

	final_t = float(t)
	final_step = int(step)

	# Loop over the other cycles
	for i in range(2, ncycles + 1):

		# Open the trajectory file
		c_ = open('cycle{}-traj.gro'.format(i), "r")

		# Get the time and simulation steps
		dt, dstep = find_dt_and_dstep("cycle{}-traj.gro".format(i))

		# Read the first line
		line = c_.readline()

		# Change t and step so VMD can read it as a single simulation
		while line:
			if "t= " in line:
				words = re.split(r'(\s+)', line)
				for j in range(len(words)):
					if words[j] == 't=':
						final_t += dt
						final_step += dstep
						words[j+2] = "{:.5f}".format(final_t)
						words[j+6] = "{}".format(final_step)
						new_line = ''.join(words)
				merged.write(new_line)
			else:
				merged.write(line)
			line = c_.readline()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Merge cycle*-traj.gro trajectory files from GROMACS trjconv format.")
	parser.add_argument("ncycles", type=int, help="the number of evaporation cycles.")

	args = parser.parse_args()

	merge_gro_files(args.ncycles)


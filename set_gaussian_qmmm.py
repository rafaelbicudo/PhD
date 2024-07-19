#!/usr/bin/env python3

"""Create Gaussian09/16 input files to run calculations according to the s-QM/MM method.

    [ ] Add an option to loop over a directory with several files.
"""


import argparse
import sys


# Functions
def parse_range(value: str):
    """Combine values from nested lists into a single list.

    Args:
        value (str): string with intervals

    Returns:
        (list): list with all values in a single list.
    """
    # Check if the input is in the form of a range (e.g., '1-100')
    if '-' in value:
        start, end = value.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        # Otherwise, it might be a single integer
        return [int(value)]


def read_gro_file(grofile: str) -> dict:
    """Extract data from Gromos87 file.

    Args:
        grofile (str): .gro file name.

    Returns:
        data (dict): data from .gro file. 
    """
    
    data = {"resnum"     : [],
            "resname"    : [],
            "atomname"   : [],
            "atomnum"    : [],
            "x_coord"    : [],
            "y_coord"    : [],
            "z_coord"    : [],
            "itpAtomNum" : []
            }

    # Read the grofile
    with open(grofile, "r") as f:
        lines = f.readlines()

        # Get the header
        data["header"] = lines[0]

        # Get the number of atoms
        data["n_atoms"] = int(lines[1].split()[0])
        
        # Get the box dimensions
        data["box_dim"] = (float(lines[-1].split()[0]), 
                           float(lines[-1].split()[0]), 
                           float(lines[-1].split()[0]))

        for line in lines[2:-1]:

            # Get the residue number
            data["resnum"].append(int(line[:5]))

            # Get the residue name
            data["resname"].append(line[5:10].split()[0])

            # Get the atom name
            data["atomname"].append(line[10:15].split()[0])

            # Get the atom number
            data["atomnum"].append(int(line[15:20]))

            # Get the atomic coordinates
            data["x_coord"].append(float(line[20:28]))
            data["y_coord"].append(float(line[28:36]))
            data["z_coord"].append(float(line[36:44]))

    # Get the itp ordering by tracking the change in residue number
    for i in range(len(data["resnum"])):
        if i == 0 or data["resnum"][i] != data["resnum"][i-1]:
            data["itpAtomNum"].append(1)
            j = 2
        else:
            data["itpAtomNum"].append(j)
            j += 1

    return data


def read_itp_file(itpfile: str) -> dict:
    """Extract data from GROMACS topology (.itp) file.

    Args:
        itpfile (str): .itp file name.

    Returns:
        data (dict): data from .itp file.
    """

    def parse_file(f) -> None:
        """Reads .itp files and parse the data.

        Args:
            f (_io.TextIOWrapper): _description_
        """
        line = f.readline()

        # Get the atomic data
        while "[ atoms ]" not in line: 
            line = f.readline()

        while line:
            line = f.readline()
            if line.strip() == "" or line.startswith("[ "):
                break 
            elif line.startswith(";"):
                pass
            else:
                words = line.split()

                # Get the atom number
                data["atomnum"].append(words[0])

                # Get the atom type
                data["atomtype"].append(words[1])

                # Get the residue name
                data["resname"].append(words[3])

                # Get the atom name
                data["atomname"].append(words[4])

                # Get the atomic charge    
                data["atomcharge"].append(float(words[6]))


    data = {"atomnum"    : [],
            "atomtype"   : [],
            "resname"    : [],
            "atomname"   : [],
            "atomcharge" : []
            }

    # Read the topology file(s)
    if isinstance(itpfile, list):
        for file in itpfile:
            with open(file, "r") as f:
                parse_file(f)

    else:
        with open(itpfile, "r") as f:
            parse_file(f)
                
    return data


def fix_resnum(
    grofile: str, 
    itpfile: str
) -> None:
    """Add the residue number to account for equal molecules 
       with different configurations. It was developed for the
       aqueous-processable polymers.

    Args:
        grofile (str): .gro file name.
        itpfile (str): .itp file name.

    Returns:
        An "updated_{grofile}" file with residues changed.
    """

    gro_data = read_gro_file(grofile)
    itp_data = read_itp_file(itpfile[0])

    # Get the number of atoms per residue from itpfile
    n_atoms_per_mol = len(itp_data["atomcharge"])

    # Get the number of molecules
    n_mols = int(gro_data["n_atoms"]/n_atoms_per_mol)

    # Create a list with updated residue numbers
    new_resnum = []
    for i in range(1, n_mols+1):
        for j in range(n_atoms_per_mol):
            new_resnum.append(i)

    # Create a file with updated residue numbers
    with open(f"updated_{grofile}", "w") as fout:
        # Write the header
        fout.write(gro_data["header"])

        # Write the amount of atoms
        fout.write(f"{gro_data['n_atoms']}\n")

        # Write the atomic data
        for i in range(len(gro_data["atomnum"])):
            new_line = "{:>5}{:<5}{:>5}{:>5}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(new_resnum[i], 
                        gro_data["resname"][i], gro_data["atomname"][i],
                        gro_data["atomnum"][i], gro_data["x_coord"][i],
                        gro_data["y_coord"][i], gro_data["z_coord"][i])

            fout.write(new_line)

        # Write the box size
        fout.write("   {}   {}   {}\n".format(gro_data['box_dim'][0], gro_data['box_dim'][1], gro_data['box_dim'][2]))


def get_QM_atoms(
    grofile: str,
    atomnums: list,
    residues: list,
    resnums: int
) -> list:
    """Get a list with atoms to be treated with QM.

    Args:
        grofile (str): .gro file name.
        itpfile (list[str]): list with .itp file names.
        atomnums (list): QM atom numbers (indexes) from the .gro file.
        residues (list): QM residues from the .gro file.
        resnums (int): number of the residue to get the QM atoms from .gro file.
                       Only used when a list of residues is provided.

    Returns:
        qm_atoms (list): list with QM atoms with atom numbers from .gro file. 
    """

    # Get the data dictionary
    gro_data = read_gro_file(grofile)

    # Checking for each combination of input to create a list with QM atoms
    if atomnums and not residues:
        qm_atoms = atomnums
        
        # Flatten the qm_atoms in case of nested lists
        if any(isinstance(i, list) for i in qm_atoms):
            qm_atoms = [atom for sublist in qm_atoms for atom in sublist]

    elif not atomnums and residues:
        print(f"Selecting residue names {residues} with residue numbers {resnums}.\n")

        qm_atoms = []
        for i in residues:
            for j in range(len(gro_data["resname"])):
                if gro_data["resname"][j] == i:
                    for k in resnums:
                        if k == gro_data["resnum"][j]:
                            qm_atoms.append(gro_data["atomnum"][j])

    else:
        print('Please provide atom numbers (from .gro file) or residues of the QM atoms.')
        sys.exit()

    return qm_atoms


def get_charge(
    itpfile: str,
    atom_name: str,
    res_name: str
) -> float:
    """Get the charge from the .itp file(s).

    Args:
        itpfile (str): .itp file name.
        atom_name (str): name of the atom.
        res_name (str): name of the residue.

    Returns:
        charge (float): charge of "atom_name".
    """

    # Get the data dictionary
    itp_data = read_itp_file(itpfile)

    # Loop over all atom names
    for i in range(len(itp_data["atomname"])):
        
        # Check for matching atom names
        if itp_data["atomname"][i] == atom_name:

            # Check for matching residue names
            if itp_data["resname"][i] == res_name:

                charge = itp_data["atomcharge"][i]
                # print(f"Charge of {charge} found in the {itpfile} topology file.")

                return charge
            
    # print(f"Couldn't find the atom in the {itpfile} topology file.")
    # sys.exit()


def get_connectivity(
    itpfile: list, 
    atomnum: int, 
    resname: str
) -> list:
    """Returns a list with bonded atoms to atomnum.

    Args:
        itpfile (list[str]): list with .itp file names.
        atomnum (int): atom number in the .itp file.
        resname (str): residue name of the corresponding atom.

    Returns:
        bonded_atoms (list): list with bonded atoms numbers (from the .itp file)
                             to the corresponding atom.
    """

    # Start the bonded atoms numbers list
    bonded_atoms = []

    # Open the .itp file(s)
    for file in itpfile:
        
        # Get the .itp data
        itp_data = read_itp_file(file)

        # Check if the residue is in the current .itp file
        res = False
        if resname in set(itp_data["resname"]):
            res = True
        
        # If not, save time by going to the next file
        if not res:
            continue

        with open(file, "r") as f:
            line = f.readline()

            # Read until the bonds block is found
            while "[ bonds ]" not in line:
                line = f.readline()

            # Reads until the file or the bonds block end
            while line:
                line = f.readline()
                if line.strip() == "" or line.startswith("[ "):
                    break 

                words = line.split()
                if str(f"{atomnum}") == words[0]: 
                    bonded_atoms.append(words[1])
                elif str(f"{atomnum}") == words[1]:
                    bonded_atoms.append(words[0])

        # Leaves the loop if the residue was found
        if res:
            break

    # Convert the bonded atom numbers into integers
    bonded_atoms = [int(x) for x in bonded_atoms]

    return file, bonded_atoms


def get_closest_atoms(
    grofile: str,
    target: int,
    cutoff: float,
    n_atoms: int
) -> list:
    """Returns a list of the n_atoms-th closest atoms.

    Args:
        grofile (str): .gro file with all atoms.
        target (int): atom number of the target atom to get.
        cutoff (float): cutoff distance (in AA).
        n_atoms (int): number of closest atoms to be returned. 

    Returns:
        closest_atoms (list): list with atom numbers of the closest atoms.
    """

    # Get the data dictionary
    gro_data = read_gro_file(grofile)

    # Define the reference coordinates
    x = gro_data["x_coord"][gro_data["atomnum"].index(target)]
    y = gro_data["y_coord"][gro_data["atomnum"].index(target)]
    z = gro_data["z_coord"][gro_data["atomnum"].index(target)]

    # Compute the distance with all atoms from .gro file
    distances = []
    for x_, y_, z_ in zip(gro_data["x_coord"], gro_data["y_coord"], gro_data["z_coord"]):
        # Compute the euclidean distance and append it to the list
        dist = ((x_-x)**2+(y_-y)**2+(z_-z)**2)**(1/2)
        distances.append(dist)

    # Create a list with atomic numbers
    atom_nums = gro_data["atomnum"].copy()

    # Sort the atomic numbers with increasing distances
    sorted_list = sorted(list(zip(distances, atom_nums)), key=lambda x: x[0])

    # Unzip the paired lists
    distances, closest_atoms = zip(*sorted_list)

    # Convert back to list
    distances = list(distances)
    closest_atoms = list(closest_atoms)

    if cutoff == 0:
        # Return the first "n_atoms" from this list
        return closest_atoms[:n_atoms], distances[:n_atoms]
    else:
        # Return the atoms closer than the "cutoff" distance
        for i in range(len(distances)):
            if distances[i] > cutoff/10:
                return closest_atoms[:i], distances[:i]


def h_link_saturation(
    grofile: str,
    itpfile: str,
    # resnames: list,
    qm_atoms: list,
    dist_scale_factor: float
) -> None:
    """Saturates the QM region with hydrogens.

    Args:
        grofile (str): .gro file with all atoms.
        itpfile (str): .itp topology file(s).
        # resnames (list): list with all resnames.
        qm_atoms (list): list with atoms to be treated with QM.
        dist_scale_factor (float): scale factor to the link atom bond distance.

    Returns:
        to_remove_num (list): list with .gro atom numbers to be removed.
        h_links (list): list with coordinates of hydrogen link atoms.
    """

    # Get the data dictionaries
    gro_data = read_gro_file(grofile)

    # Initialize the list of atoms to be removed
    to_remove_num = []
    link_coords = []

    # Flatten the qm_atoms in case of nested lists
    if any(isinstance(i, list) for i in qm_atoms):
        qm_atoms = [atom for sublist in qm_atoms for atom in sublist]

    # Loop over QM atoms
    for i in qm_atoms:

        # Get the resname and the atom number (in the .itp file) of each QM atom
        itpAtomNum = gro_data["itpAtomNum"][gro_data["atomnum"].index(i)]
        resname = gro_data["resname"][gro_data["atomnum"].index(i)]

        # Get the bonded atoms (.itp file's atom number) to the QM atom
        _, bonded_atoms = get_connectivity(itpfile, itpAtomNum, resname)

        # Loop over the 20 closest atoms 
        closest_atoms, _ = get_closest_atoms(grofile, i, cutoff=0, n_atoms=20)

        for atom in closest_atoms:

            # Get the resname and the atom number (in the .itp file) of the neighbor
            _itpAtomNum = gro_data["itpAtomNum"][gro_data["atomnum"].index(atom)]

            # Check if the neighbor is bonded to the QM atom and 
            # belongs to different residues to find the atoms to be removed
            if _itpAtomNum in bonded_atoms and atom not in qm_atoms:
                to_remove_num.append(atom)

                # Get the atomic coordinates of both QM and classical atoms
                x = gro_data["x_coord"][gro_data["atomnum"].index(i)]
                y = gro_data["y_coord"][gro_data["atomnum"].index(i)]
                z = gro_data["z_coord"][gro_data["atomnum"].index(i)]

                x_ = gro_data["x_coord"][gro_data["atomnum"].index(atom)]
                y_ = gro_data["y_coord"][gro_data["atomnum"].index(atom)]
                z_ = gro_data["z_coord"][gro_data["atomnum"].index(atom)]

                # Determine the link atom coordinates
                x_l = x + dist_scale_factor * (x_ - x)
                y_l = y + dist_scale_factor * (y_ - y)
                z_l = z + dist_scale_factor * (z_ - z)

                link_coords.append((x_l, y_l, z_l))

    return to_remove_num, link_coords


def get_charge_shifts(
    grofile: str,
    itpfile: str,
    qm_atoms: list,
    to_remove: list,
    n_neighbors: int,
    cutoff: float
) -> float | list:
    """Compute the charges to neutralize the system.

    Args:
        grofile (str): .gro file with all atoms.
        itpfile (str): .itp topology file(s).
        qm_atoms (list): list with atoms to be treated with QM.
        to_remove (list): list with atoms to be removed.
        n_neighbors (int): number of closest neighbors to redistribute the charge.
        cutoff (float): cutoff radius to redistribute the charge.
    Returns:
        qm_charge_shift (float): charge shift per atom from the QM atoms.
        neigh_sums (list): list with neighbor atoms.
        neigh_charge_shifts (list): list with charges to be added to the neighbor atoms.
    """

    # Initialize the variables
    qm_charge = 0
    neigh_nums = []
    neigh_charge_shifts = []

    # Get the data dictionary
    gro_data = read_gro_file(grofile)

    # Flatten the qm_atoms in case of nested lists
    if any(isinstance(i, list) for i in qm_atoms):
        qm_atoms = [atom for sublist in qm_atoms for atom in sublist]

    # Get the total charge of the QM atoms
    for j in qm_atoms:
        # Get the atom and residue name of each to be removed
        atomname = gro_data["atomname"][gro_data["atomnum"].index(j)]
        resname = gro_data["resname"][gro_data["atomnum"].index(j)]

        # Loop over the .itp files
        for file in itpfile:
            charge = get_charge(file, atomname, resname)
            if charge:
                qm_charge += charge 
                break

    # Loop over the atoms to be removed
    for i in to_remove:
        # Get the atom and residue name of each to be removed
        atomname = gro_data["atomname"][gro_data["atomnum"].index(i)]
        resname = gro_data["resname"][gro_data["atomnum"].index(i)]

        # Gets the charge of the atom
        for file in itpfile:
            # Search for the charge
            charge = get_charge(file, atomname, resname)
            # Breaks the loop if the charge is found
            if charge:
                break

        # Get all the atoms inside a sphere with "cutoff" radius
        closest_atoms, distances = get_closest_atoms(grofile, i, cutoff, n_neighbors)

        # Get the the "n_neighbors" closest atoms
        # [1:n_neighbors+1] to remove the distance with respect to itself
        closest_atoms = closest_atoms[1:n_neighbors+1]
        distances = distances[1:n_neighbors+1]

        # Add the closest atoms to the list with neighbors
        neigh_nums.extend(closest_atoms)

        # Equally redistribute the charge over the "n_neighbors" atoms
        for i in closest_atoms:
            neigh_charge_shifts.append(-charge/len(closest_atoms))

    # Compute the overall charge shift
    qm_charge_shift = qm_charge/(gro_data["n_atoms"]-len(to_remove)-len(qm_atoms))

    return qm_charge_shift, neigh_nums, neigh_charge_shifts


def write_gaussian_input(
    grofile: str,
    itpfile: str,
    qm_atoms: list,
    to_remove: list,
    link_coords: list,
    n_neighbors: int,
    cutoff: float,
    keywords: str,
    charge: int,
    spin_mult: int,
    output: str,
    test: bool,
) -> None:
    """Write the Gaussian09/16 input file.

    Args:
        grofile (str): .gro file with all atoms.
        itpfile (str): .itp topology file(s).
        qm_atoms (list): list with atoms to be treated with QM.
        to_remove (list): list with atoms to be removed.
        link_coords (list): list with link atoms coordinates.
        n_neighbors (int): number of closest neighbors to redistribute the charge.
        cutoff (float): cutoff radius to redistribute the charge.
        keywords (str): calculation keywords (e.g. HF/STO-3G Charge)
        charge (int): system's total charge.
        spin_mult (int): system's spin multiplicity.
        output (str): name of the output file.
        test (bool): write point charges as bismuth atoms for visualization.
    """

    # Get the data dictionaries
    gro_data = read_gro_file(grofile)

    # Get the charge shifts
    qm_shift, neigh_nums, neigh_shifts = get_charge_shifts(grofile, itpfile, qm_atoms, to_remove, n_neighbors, cutoff)

    # Write the coordinates in the Gaussian input file
    with open(f"{output}", "w") as fout:

        # Write the header
        fout.write(f"chk={output.split('.')[0]}.chk \n")
        fout.write(f"#p {' '.join(keywords)} \n\n")
        fout.write(f"QM calculation with point charges \n\n")
        fout.write(f"{charge} {spin_mult}\n")

        # Write the QM atoms
        for j in range(len(gro_data["resname"])):
            # Write the QM atoms
            if gro_data["atomnum"][j] in qm_atoms:
                
                # Check if the second letter of atom name is lower case
                if len(gro_data["atomname"][j])>1 and gro_data["atomname"][j][1].islower():
                    atom_name = gro_data["atomname"][j][:2]
                else:
                    atom_name = gro_data["atomname"][j][:1]
            
                # Write the atom type and coordinates
                fout.write("{}\t{:>.3f}\t{:>.3f}\t{:>.3f}\n".format(
                            atom_name, gro_data["x_coord"][j]*10, 
                            gro_data["y_coord"][j]*10, gro_data["z_coord"][j]*10))
        
        # Write the QM link atoms
        for j in link_coords:
            fout.write("H\t{:>.3f}\t{:>.3f}\t{:>.3f}\n".format(j[0]*10, j[1]*10, j[2]*10))

        # Write the blank line required by Gaussian
        if not test:
            fout.write("\n")
            
        # Write the remaining atoms as point charges
        for j in range(len(gro_data["resname"])):
            # Write the other atoms as point charges
            if (gro_data["atomnum"][j] not in qm_atoms
                and gro_data["atomnum"][j] not in to_remove):

                # Get the charge
                charge = get_charge(itpfile, gro_data["atomname"][j], gro_data["resname"][j])

                # Redistribute the charges to neutralize the system
                if gro_data["atomnum"][j] in neigh_nums:
                    charge += qm_shift + neigh_shifts[neigh_nums.index(gro_data["atomnum"][j])]
                else:
                    charge += qm_shift

                # Boolean variable to check if partial charges are correctly placed
                if test:
                    # Write the partial charges as bismuth atoms
                    fout.write("Bi\t{:>.3f}\t{:>.3f}\t{:>.4f}\n".format(
                                gro_data["x_coord"][j]*10, 
                                gro_data["y_coord"][j]*10,
                                gro_data["z_coord"][j]*10))
                else:
                    # Write the coordinates and partial charge
                    fout.write("{:>.3f}\t{:>.3f}\t{:>.3f}\t{:>.4f}\n".format(
                                gro_data["x_coord"][j]*10, 
                                gro_data["y_coord"][j]*10,
                                gro_data["z_coord"][j]*10, 
                                charge))

        # Write the final blank line required by Gaussian
        fout.write("\n")


def check_total_charge(file: str, test: bool) -> None:
    """Add the spurious charge to the last partial charge.  

    Args:
        file (str): name of the Gaussian file.
        test (bool): avoid searching for charges during visualization tests.
    Returns:
        A "fixed_{file}" file.
    """

    with open(file, "r") as f:
        line = f.readline()

        # Read lines until the third blank line is found
        if test:
            print("No partial charges when --test is set.")
        else:
            i = 0
            while i < 3:
                line = f.readline()
                if line.strip() == "":
                    i += 1

        # Loop over the partial charges
        while line:
            line = f.readline()
            
            # Get the total charge
            total_charge = 0
            if len(line.split()) == 4:
                total_charge += float(line.split()[3])
    
    # Neutralize the total charge if it isn't zero
    if not test:
        if round(total_charge, 3) == 0:
            print(f"Total charge is {total_charge:>.3f}. No need for fixing numerical fluctuations.")
        else:
            with open(file, "r") as f:
                lines = f.readlines()
                last_line = lines[-2].split()
                last_line[-1] = str(float(last_line[-1]) - float(total_charge))
                lines[-2] = '\t'.join(last_line) + "\n"

            with open(f"fixed_{file}", "w") as fout:
                for line in lines:
                    fout.write(line)

            print(f"Total charge is {total_charge:>.3f}. Charge {-total_charge:>.3f} was added to the last partial charge.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract configurations from Gromos87 (.gro) files.")
    
    parser.add_argument("grofile", help="reference .gro configuration file.", type=str)
    parser.add_argument("itpfile", help="topology file(s) (.itp) from GROMACS.", nargs="+", type=str, default=[])
    parser.add_argument("--atomnums", "-an", help="list of atom numbers treated with QM (e.g. 1-10 22 82-93).", nargs="+", type=parse_range, default=[])
    parser.add_argument("--residues", "-res", help="list of residues treated with QM.", nargs="+", type=str, default=[])
    parser.add_argument("--resnums", "-rn", help="number of the residue(s) to be treated with QM.", nargs="+", type=int, default=[1])
    parser.add_argument("--link_scale_factor", "-sf", help="link atom distance scale factor.", type=float, default=0.71)
    parser.add_argument("--n_neighbors", "-nn", help="number of closest neighbors to redistribute the charge.", type=int, default=3)
    parser.add_argument("--cutoff", "-cut", help="cutoff radius (in AA) to redistribute the charge.", type=float, default=5.0)
    parser.add_argument("--fix_resnum", "-fr", help="fix the residue numbering (for polymatic example).", action="store_true")
    # parser.add_argument("--configs_dir", "-dir", help="path to the directory with .gro configurations.", type=str, default=".")
    parser.add_argument("--keywords", "-k", help="Gaussian keywords for the calculation.", nargs="+", type=str, default=["B3LYP/6-31G(d,p) Charge"])
    parser.add_argument("--charge", "-c", help="total charge of the system.", type=int, default=0)
    parser.add_argument("--spin_multiplicity", "-ms", help="spin multiplicity of the system.", type=int, default=1)
    parser.add_argument("--output", "-o", help="name of the output file", type=str, default="calc.com")
    parser.add_argument("--test", help="If True, set partial charges as bismuth atoms for visualization.", action="store_true", default=False)

    args = parser.parse_args()

    # Hardcoded section to set residue numbers for the PQx8O-T polymatic example
    if args.fix_resnum:
        fix_resnum(args.grofile, args.itpfile)
        qm_atoms = get_QM_atoms(f"updated_{args.grofile}", args.atomnums, args.residues, args.resnums)

    else:
        # Get the .gro atom numbers of QM atoms
        qm_atoms = get_QM_atoms(args.grofile, args.atomnums, args.residues, args.resnums)

    # Get the .gro atom numbers of atoms to be removed and the link atoms coordinates
    to_remove_num, link_coords = h_link_saturation(
        args.grofile, 
        args.itpfile, 
        # args.residues, 
        qm_atoms,
        args.link_scale_factor
    )

    write_gaussian_input(
        args.grofile, 
        args.itpfile,
        qm_atoms,
        to_remove_num,
        link_coords,
        args.n_neighbors,
        args.cutoff,
        args.keywords, 
        args.charge, 
        args.spin_multiplicity, 
        args.output,
        args.test,
    )

    check_total_charge(args.output, args.test)


if __name__ == '__main__':
    main()
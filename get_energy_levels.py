#!/usr/bin/env python3

"""Script to generate energy levels.
"""

import numpy as np # type: ignore
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib import lines as mlines

import argparse


def plot_energy_levels(
    mols: list,
    red_pot: list,
    oxi_pot: list,
    homo_en: list,
    lumo_en: list,
    ea: list,
    ip: list,
    unit: str,
    bar: bool,
    vals: bool,
    title: str
):
    
    # Define the colors
    colors = [colormaps["plasma"](i) for i in np.linspace(0, 4, 20)]

    # Create a plot
    fig, ax = plt.subplots()
            
    # Define the positions of each molecule bar plot
    x_pos = np.arange(len(mols))

    # Plot the potentials
    for x, oxi, red in zip(x_pos, oxi_pot, red_pot):
        if bar:
            # Plot a vertical bar representing the difference between oxidation and reduction potential
            ax.bar(x, red-oxi, bottom=oxi, width=0.5, color='white', edgecolor=colors[x], linewidth=2, alpha=0.7)
        else:
            # Plot the horizontal lines
            ax.hlines(oxi, x-0.3, x+0.3, color=colors[x], linewidth=2)
            ax.hlines(red, x-0.3, x+0.3, color=colors[x], linewidth=2)

        if vals:
            # Annotate the potential values
            ax.text(x, oxi-0.1, f'{oxi:.2f}', ha='center', va='top', color='black', fontsize=12)
            ax.text(x, red+0.1, f'{red:.2f}', ha='center', va='bottom', color='black', fontsize=12)

    # Plot the orbital energies
    for x, homo, lumo in zip(x_pos, homo_en, lumo_en):
        if bar:
            # Plot a vertical bar
            ax.bar(x-0.15, lumo-homo, bottom=homo, width=0.5, color='white', edgecolor=colors[x], linewidth=2, alpha=0.7, linestyle='--')
        else:
            # Plot the horizontal lines
            ax.hlines(homo, x-0.3, x+0.3, colors=colors[x], linewidth=2, linestyle='dashed')
            ax.hlines(lumo, x-0.3, x+0.3, colors=colors[x], linewidth=2, linestyle='dashed')
        
        if vals:
            # Annotate the values
            ax.text(x-0.15, homo-0.1, f'{homo:.2f}', ha='center', va='top', color='black', fontsize=12)
            ax.text(x-0.15, lumo+0.1, f'{lumo:.2f}', ha='center', va='bottom', color='black', fontsize=12)

    # Plot the EA and IP
    for x, i, j in zip(x_pos, ip, ea):
        if bar:
            # Plot a vertical bar
            ax.bar(x+0.15, j-i, bottom=i, width=0.5, color='white', edgecolor=colors[x], linewidth=2, alpha=0.7, linestyle=':')

        else:
            # Plot the horizontal lines
            ax.hlines(i, x-0.3, x+0.3, colors=colors[x], linewidth=2, linestyle='dotted')
            ax.hlines(j, x-0.3, x+0.3, colors=colors[x], linewidth=2, linestyle='dotted')

        if vals:
            # Annotate the values
            ax.text(x+0.15, i-0.1, f'{i:.2f}', ha='center', va='top', color='black', fontsize=12)
            ax.text(x+0.15, j+0.1, f'{j:.2f}', ha='center', va='bottom', color='black', fontsize=12)
   
    # Set plot details
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mols)#, ha='right', rotation=45)
    ax.set_ylabel(f"Energy levels ({unit})", fontsize=14)
    ax.set_ylim(np.min(homo_en + oxi_pot + ip)*1.1, np.max(lumo_en + red_pot + ea)*0.8+1.0)
    ax.tick_params(labelsize=12, width=1.25)
    
    # Set the legend
    dashed_leg = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2)
    dotted_leg = mlines.Line2D([], [], color='black', linestyle=':', linewidth=2)
    solid_leg = mlines.Line2D([], [], color='black', linewidth=2)

    ax.legend(
        handles=[solid_leg, dashed_leg, dotted_leg],  # List of proxy artists
        labels=[r"$\phi_{\rm oxi/red}$", r"HOMO/LUMO", r"IP/EA"],  # Corresponding labels
        frameon=False,
        loc='upper center',
        ncols=3,
        fontsize=14
    )

    # Hide top, right, and bottom spines (keep only the left spine, which is the y-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1.25)

    # Hide x-axis ticks and labels
    ax.xaxis.set_ticks_position('none')

    # Set the title
    if title != "notitle":
        ax.set_title(f"{title}", fontsize=14)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot energy levels")
    parser.add_argument("--molecules", "-mols", help="name of the molecules", nargs='+', type=str, required=True)
    parser.add_argument("--red_pot", "-red", help="reduction potentials", nargs="+", type=float, default=[])
    parser.add_argument("--oxi_pot", "-oxi", help="oxidation potentials", nargs="+", type=float, default=[])
    parser.add_argument("--homo_en", "-homo", help="homo energy levels", nargs="+", type=float, default=[])
    parser.add_argument("--lumo_en", "-lumo", help="lumo energy levels", nargs="+", type=float, default=[])
    parser.add_argument("-ea", help="electron affinities", nargs="+", type=float, default=[])
    parser.add_argument("-ip", help="ionization potentials", nargs="+", type=float, default=[])
    parser.add_argument("--unit", "--un", help="energy units (default: eV)", type=str, default="eV")
    parser.add_argument("--bar", help="If True, shows a bar between energy levels", action="store_true", default=False)
    parser.add_argument("--vals", help="If True, shows the values of each energy level", action="store_true", default=False)
    parser.add_argument("--title", "-t", help="plot title (default = none)", type=str, default="notitle")

    args = parser.parse_args()

    plot_energy_levels(
        args.molecules, 
        args.red_pot, 
        args.oxi_pot, 
        args.homo_en,
        args.lumo_en,
        args.ea,
        args.ip,
        args.unit,
        args.bar,
        args.vals,
        args.title
    )
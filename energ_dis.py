#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Read in the csv file using Pandas
data = pd.read_csv('data_nx.csv')#, header=None, squeeze=True)

ip_data = data['E_{HOMO} (eV)'].to_numpy()
ea_data = data['E_{LUMO} (eV)'].to_numpy()
ip_data = np.array([x*(-1) for x in ip_data])
ea_data = np.array([x*(-1) for x in ea_data])

z_score = False

if z_score:
	# Remove outliers using z-score
	z_scores = (ip_data - np.mean(ip_data)) / np.std(ip_data)

	# Create a boolean mask to remove outliers
	mask = np.where(abs(z_scores) < 3)

	# Apply the mask to the original data
	ip_data_no_outliers = ip_data[mask]

	# Same for EA
	z_scores2 = (ea_data - np.mean(ea_data)) / np.std(ea_data)
	mask2 = np.where(abs(z_scores2) < 3)
	ea_data_no_outliers = ea_data[mask2]

else:
	# Remove outliers using boxplot

	# calculate the interquartile range
	q1_ip, q3_ip = np.percentile(ip_data, [25, 75])
	q1_ea, q3_ea = np.percentile(ea_data, [25, 75])
	iqr_ip = q3_ip - q1_ip
	iqr_ea = q3_ea - q1_ea

	# calculate the lower and upper bounds for outliers
	lower_bound_ip = q1_ip - 1.5*iqr_ip
	upper_bound_ip = q3_ip + 1.5*iqr_ip
	lower_bound_ea = q1_ea - 1.5*iqr_ea
	upper_bound_ea = q3_ea + 1.5*iqr_ea

	# filter out the outliers using Boolean indexing
	ip_data_no_outliers = ip_data[(ip_data >= lower_bound_ip) & (ip_data <= upper_bound_ip)]
	ea_data_no_outliers = ea_data[(ea_data >= lower_bound_ea) & (ea_data <= upper_bound_ea)]

# Compute the outliers
retirados = np.setdiff1d(ip_data, ip_data_no_outliers)
retirados2 = np.setdiff1d(ea_data, ea_data_no_outliers)

# Create a figure and define the layout
fig, ax = plt.subplots(2, 1, figsize=(5, 6))#, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot the IP histogram
n, bins, patches = ax[0].hist(ip_data, bins=30, density=True, facecolor='#2ab0ff', edgecolor='tab:blue', alpha=0.8)
														#, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.plasma(n[i]/max(n)))

ax[0].set_ylabel('Frequency')
ax[0].yaxis.set_label_coords(-0.1, -0.1)

# Fit a Gaussian distribution
(mu, sigma) = norm.fit(ip_data_no_outliers)
y = norm.pdf(bins, mu, sigma)
ax[0].plot(bins, y, 'r--', linewidth=2, label=r'Gaussian fit ($\mu$={}, $\sigma$={})'.format(round(mu, 3), round(sigma, 3)))
ax[0].scatter(retirados, np.full(len(retirados), 1), marker='+', color='black', label='Outliers', linewidth=1)

# Plot the boxplot
ax[0].boxplot(ip_data, vert=False, notch=True, flierprops=dict(marker='+', markeredgecolor='black', alpha=1), sym='')
ax[0].set_xlabel('IP (eV)')
ax[0].set_yticks([1, 2, 3, 4, 5, 6])
ax[0].set_yticklabels(["1", "2", "3", "4", "5", "6"])
ax[0].set_ylim([0, 6])
ax[0].legend(frameon=False)

# Plot the EA histogram
n2, bins2, patches2 = ax[1].hist(ea_data, bins=30, density=True, facecolor='#2ab0ff', edgecolor='tab:blue', alpha=0.8)

n2 = n2.astype('int')
for i in range(len(patches2)):
    patches2[i].set_facecolor(plt.cm.plasma(n2[i]/max(n2)))

(mu2, sigma2) = norm.fit(ea_data_no_outliers)
y2 = norm.pdf(bins2, mu2, sigma2)
ax[1].boxplot(ea_data, vert=False, notch=True, flierprops=dict(marker='+', markeredgecolor='black', alpha=1), sym='')
ax[1].plot(bins2, y2, '--', color='r', linewidth=2, label=r'Gaussian fit ($\mu$={}, $\sigma$={})'.format(round(mu2, 3), round(sigma2, 3)))
ax[1].scatter(retirados2, np.full(len(retirados2), 1), marker='+', color='black', label='Outliers', linewidth=1)
ax[1].set_xlabel('EA (eV)')
ax[1].set_yticks([1, 2, 3, 4, 5, 6, 7])
ax[1].set_yticklabels(["1", "2", "3", "4", "5", "6", "7"])
ax[1].set_ylim([0, 7])
ax[1].legend(frameon=False)

# Show the plot
plt.tight_layout()
plt.yticks(visible=True)
plt.show()

# fig.savefig('/Users/Rafael/Coisas/Doutorado/Publicação/desordem_artigo3.png', bbox_inches='tight', format='png', dpi=600)

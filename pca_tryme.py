"""

Filename: pca_tryme

Purpose: Test the pca_tools module

Author: John Weaver
Date created: 02.05.18
Possible problems:
1. Can't run single eigenspectra due to print statements

"""
# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------
import pygappy.pca_projection as pca
import pygappy.visualization as pcaplot

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Number of eigenspectra to use ( >2 & <<440)
Nspec = 10

# Number of test spectra to generate
Ntry = 100

# Verbosity
verbose = True

# Noise
noise = True

# Norm factor
appnorm = 1.

# PCs to be shown
Nshow = 0

# Seed
np.random.seed(0)

# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------

# Get eigensystem
ewave, espec, emean, evar = pca.eigenbasis()
espec = espec[:Nspec]

# Generate test spectra
test_pcs = np.random.normal(0, 0.5, (Ntry, Nspec))
test_pcs[:, 0] = np.random.normal(-6, 2, Ntry)
test_pcs[:, 1] = np.random.normal(0, 2, Ntry)
test_spec = test_pcs @ espec

# Generate error array
error = np.ones(440)

# Create figure
fig1, ax1 = plt.subplots(ncols=1, nrows=2)

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------

# Loop over test spectra
for i, spec in enumerate(test_spec):

    # Prepare data
    data = emean + spec
    if noise:
        data = data + np.random.normal(0, 0.02, 440)
    data = appnorm * (data / np.mean(data))

    if verbose:
        print(f'\n{i+1}/{Ntry} -----------------------------------')
        x, y, z = test_pcs[i, :3]
        print(f'    first 3 PCs: {x:2.1f} {y:2.1f} {z:2.1f}')

    # Get GappyPCA results
    gpcs, cov = pca.pca_gappy(
        data, error, espec, emean, cov=True, verbose=verbose)

    # Check them
    if verbose:
        x, y, z = gpcs[:3]
        print(
            f'    GappyPCA: {x:2.1f} {-y:2.1f} {z:2.1f}'
        )

    # Plot them
    pcaplot.model(data, ewave, gpcs, espec, ax=ax1[0], mean=emean,
                   nshow=Nshow)

    # Get NormGappyPCA results
    npcs, norm, cov = pca.pca_normgappy(
        data, error, espec, emean, cov=True, verbose=verbose)

    # Check them
    if verbose:
        x, y, z = npcs[:3]
        print(
            f'    NormGappyPCA: {x:2.1f} {-y:2.1f} {z:2.1f} | N={norm:2.1f}'
        )

    # Plot them
    pcaplot.model(data, ewave, npcs, espec, ax=ax1[1], mean=emean,
                   norm=norm, nshow=Nshow)

    # ID the axes
    ax1[0].text(
        0.1, 0.1, 'GappyPCA', color='darkred', transform=ax1[0].transAxes)
    ax1[1].text(
        0.1,
        0.1,
        'NormGappyPCA',
        color='royalblue',
        transform=ax1[1].transAxes)

    # Next
    plt.pause(0.001)
    if (i + 1) != Ntry:
        input('\nNext? ')
        ax1[0].clear()
        ax1[1].clear()
    else:
        print('\nEnd.')

# End Script

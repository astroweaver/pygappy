# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----


Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
def pca_plot(data, wave, pcs, espec, mean=None, norm=1.0, nshow=3, ax=None):
    """
    Plots the output of a PCA analysis.

    Required keyword arguments:
    data -- the input data array
    wave -- the wavelength of the data
    pcs -- the principle component array
    espec -- the eigensystem

    Optional keyword arugments:
    mean -- eigensystem mean, required for NormGappyPCA plots (default None)
    norm -- output norm, required for NormGappyPCA plots (default 1.0)
    nshow -- Number of scaled eigenspectra to show (default 3)
    ax -- can pass in existing axis object (defautl None)

    Parameters
    ----------
    data : ndarray
        1D spectrum with 'float' type.
    wave : ndarray
        1D array of wavelengths.
    pcs : ndarray
        1D array of principal components.
    espec : ndarray
        2D array of eiegenspectra.
    mean : float, optional
        1D mean spectrum, required for normgappy pcs.
    norm : float, optional
        Normalization constant, required for normgappy pcs.
    nshow : float, optional
        Number of eigenspectra to plot.
        Default is '3'.
    ax : :obj:axes, optional
        Existing ax object to draw on.
        Default is 'None'
    """

    # deal with mean being None
    if mean is None:
        mean = np.zeros(len(espec[0]))

    # reconstruction
    pcs_nshow = norm * (mean + pcs[:nshow] @ espec[:nshow])
    yp = norm * (mean + pcs[:, None] * espec)
    precon = norm * (mean + pcs @ espec)

    # setup figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # plot stuff
    ax.plot(wave, data, c='k', alpha=1)
    ax.plot(wave, precon, c='r', ls='dashed', alpha=1)

    colors = ('darkred', 'green', 'royalblue')[:nshow]
    for i in np.arange(nshow):
        ax.plot(wave, yp[i], c=colors[i], alpha=0.5)

    for i, color in enumerate(colors):
        ax.text(
            0.75 + (i * 0.05),
            0.15,
            s='{:2.1f}'.format(pcs[i]),
            color=color,
            transform=ax.transAxes,
            horizontalalignment='right')

    ax.set_xlim([wave.min(), wave.max()])
    ax.set_xlabel('Wavelength ($\AA$)', fontsize=15)
    ax.set_ylabel('Normalised Flux', fontsize=15)

    if ax is None:
        return fig, ax
    else:
        return ax

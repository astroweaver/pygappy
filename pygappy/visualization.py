# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Visulization tools to examine output from pca_projection routines.

Regions are defined from the SDSS/MaNGA MPL-6 sample by K. Rowlands (priv. comm.)

Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
def model(data, wave, pcs, espec, error=None, mean=None, norm=1.0,
          nshow=3, show_labels=True, ax=None, normalize=False):
    """
    Plots the output model fit of a PCA analysis.

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
    error: ndarray, optional
        1D error spectrum.
        Required for windows to work.
    mean : float, optional
        1D mean spectrum, required for normgappy pcs.
    norm : float, optional
        Normalization constant, required for normgappy pcs.
    nshow : float, optional
        Number of eigenspectra to plot.
        Default is '3'.
    show_labels: bool, optional
        Turns on/off default x/y labels.
        Default is 'True'.
    normalize: bool, optional
        Turns on/off scaling normalization using norm.
        Default is 'False'.
    ax : :obj:axes, optional
        Existing ax object to draw on.
        Default is 'None'

    Returns
    -------
    ax : axes-object
        Axes containing the model plot.
    fig : figure-object, optional
        If enabled, figure containing the axes.
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

    if normalize:
        data = data / norm
        precon = precon / norm
        yp = yp / norm

    # plot stuff
    ax.plot(wave, data, c='k', alpha=1)
    ax.plot(wave, precon, c='r', ls='solid', alpha=1)

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
    if show_labels:
        ax.set_xlabel('Wavelength ($\AA$)', fontsize=15)
        ax.set_ylabel('Normalised Flux', fontsize=15)

    if ax is None:
        return fig, ax
    else:
        return ax


def pc_plane(logmass=1., ax=None, monochromatic=False):
    """
    Returns an ax object with PC1/2 plane patches.

    Parameters
    ----------
    logmass : float, optional
        Log10 stellar mass, used to select suitable region patches.
        Default is '1.', for low-mass.
    ax : axes-object, optional
        Existing axes object to populate.
        Default is 'None'.
    monochromatic : bool, optional
        If enabled, only shows region boundaries, not colors.
        Default is 'False'.

    Returns
    -------
    ax : axes-object
        Axes containing the model plot.
    fig : figure-object, optional
        If enabled, figure containing the axes.
    """

    # Make axis
    if ax is None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.15)

    # Decide mass range
    if (logmass < 10):
        pca_patches = get_patches_lowmass()

    else:
        pca_patches = get_patches_highmass()

    for patch_verts in pca_patches:
        facecolor = patch_verts[1]
        if monochromatic:
            facecolor = "white"
        patch = patches.PathPatch(
            Path(patch_verts[0]),
            facecolor=facecolor,
            edgecolor='k',
            lw=1,
            alpha=0.2)
        ax.add_patch(patch)

    ax.set(xlim=(-7.1, 2), ylim=(-2.3, 2), aspect='auto')
    ax.set_xlabel('PC1 (4000$\AA$)', fontsize=15)
    ax.set_ylabel('PC2 (Excess H$\delta$ Abs.)', fontsize=15)

    if ax is None:
        return fig, ax
    else:
        return ax


def get_patches_lowmass():
    """
    Patches for PCA-defined regions calibrated with MPL-6.
    Suitable for low-z, with Log10(M*) < 10.

    Parameters
    ----------
    None

    Returns
    -------
    pca_patches_lowmass : ndarray
        Series of patches for low-mass objects.

    """

    vertices_lowmass = {
        "psb_cut":0.39, \
        "sb_vert1":-3.0, \
        "sf_vert1":-5.8, \
        "sf_vert2":-5.5, \
        "sf_vert3":-2.2, \
        "sf_vert4":-3.2, \
        "green_vert1": -0.2, \
        "green_vert2": -2.1, \
        "junk_y_lower": -1.4, \
        "junk_y_lower2": -3., \
        "junk_y_upper": 2., \
        "left_cut": -7.1, \
        "right_cut": 2., \
        }
    pca_patches_lowmass = []
    # SF
    sf_verts = [
        (vertices_lowmass["sf_vert1"],
         vertices_lowmass["junk_y_lower"]),  # left, bottom
        (vertices_lowmass["sf_vert2"],
         vertices_lowmass["psb_cut"]),  # left, top
        (vertices_lowmass["sf_vert3"],
         vertices_lowmass["psb_cut"]),  # right, top
        (vertices_lowmass["sf_vert4"],
         vertices_lowmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_lowmass.append([sf_verts, 'blue'])

    # PSB
    psb_verts = [
        (vertices_lowmass["left_cut"],
         vertices_lowmass["psb_cut"]),  # left, bottom
        (vertices_lowmass["left_cut"],
         vertices_lowmass["junk_y_upper"]),  # left, top
        (vertices_lowmass["green_vert1"] - 2.,
         vertices_lowmass["junk_y_upper"]),  # right, top
        (vertices_lowmass["green_vert1"],
         vertices_lowmass["psb_cut"])  # right, bottom
    ]
    pca_patches_lowmass.append([psb_verts, 'purple'])

    # SB
    sb_verts = [
        (vertices_lowmass["left_cut"],
         vertices_lowmass["sb_vert1"]),  # left, bottom
        (vertices_lowmass["left_cut"],
         vertices_lowmass["psb_cut"]),  # left, top
        (vertices_lowmass["sf_vert2"],
         vertices_lowmass["psb_cut"]),  # right, top
        (vertices_lowmass["sf_vert1"] - 0.3,
         vertices_lowmass["sb_vert1"])  # right, bottom
    ]
    pca_patches_lowmass.append([sb_verts, 'yellow'])

    # Green valley
    green_verts = [
        (vertices_lowmass["sf_vert4"],
         vertices_lowmass["junk_y_lower"]),  # left, bottom
        (vertices_lowmass["sf_vert3"],
         vertices_lowmass["psb_cut"]),  # left, top
        (vertices_lowmass["green_vert1"],
         vertices_lowmass["psb_cut"]),  # right, top
        (vertices_lowmass["green_vert2"],
         vertices_lowmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_lowmass.append([green_verts, 'green'])

    # Red
    red_verts = [
        (vertices_lowmass["green_vert2"],
         vertices_lowmass["junk_y_lower"]),  # left, bottom
        (vertices_lowmass["green_vert1"] + 1.7,
         vertices_lowmass["junk_y_upper"]),  # left, top
        (vertices_lowmass["right_cut"],
         vertices_lowmass["junk_y_upper"]),  # right, top
        (vertices_lowmass["right_cut"],
         vertices_lowmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_lowmass.append([red_verts, 'red'])

    # Junk
    junk_verts = [
        (vertices_lowmass["sf_vert1"] - 0.3,
         vertices_lowmass["junk_y_lower2"]),  # left, bottom
        (vertices_lowmass["sf_vert2"] - 0.3,
         vertices_lowmass["junk_y_lower"]),  # left, top
        (vertices_lowmass["right_cut"],
         vertices_lowmass["junk_y_lower"]),  # right, top
        (vertices_lowmass["right_cut"],
         vertices_lowmass["junk_y_lower2"])  # right, bottom
    ]
    pca_patches_lowmass.append([junk_verts, 'grey'])

    return pca_patches_lowmass


def get_patches_highmass():
    """
    Patches for PCA-defined regions calibrated with MPL-6.
    Suitable for low-z, with Log10(M*) > 10.

    Parameters
    ----------
    None

    Returns
    -------
    pca_patches_highmass : ndarray
        Series of patches for high-mass objects.

    """

    vertices_highmass = {
    "psb_cut":0.39, \
    "sb_vert1":-1.9, \
    "sf_vert1":-5.3-0.07, \
    "sf_vert2":-5.0-0.07, \
    "sf_vert3":-2.2-0.07, \
    "sf_vert4":-3.2-0.07, \
    "green_vert1": -0.3-0.07, \
    "green_vert2": -2.0-0.07, \
    "junk_y_lower": -1.2, \
    "junk_y_lower2": -3., \
    "junk_y_upper": 2., \
    "left_cut": -7.1, \
    "right_cut": 2.-0.07, \
    }

    # PCA Paths
    pca_patches_highmass = []
    # SF
    sf_verts = [
        (vertices_highmass["sf_vert1"],
         vertices_highmass["junk_y_lower"]),  # left, bottom
        (vertices_highmass["sf_vert2"],
         vertices_highmass["psb_cut"]),  # left, top
        (vertices_highmass["sf_vert3"],
         vertices_highmass["psb_cut"]),  # right, top
        (vertices_highmass["sf_vert4"],
         vertices_highmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_highmass.append([sf_verts, 'blue'])

    # PSB
    psb_verts = [
        (vertices_highmass["left_cut"],
         vertices_highmass["psb_cut"]),  # left, bottom
        (vertices_highmass["left_cut"],
         vertices_highmass["junk_y_upper"]),  # left, top
        (vertices_highmass["green_vert1"] - 2.,
         vertices_highmass["junk_y_upper"]),  # right, top
        (vertices_highmass["green_vert1"],
         vertices_highmass["psb_cut"])  # right, bottom
    ]
    pca_patches_highmass.append([psb_verts, 'purple'])

    # SB
    sb_verts = [
        (vertices_highmass["left_cut"],
         vertices_highmass["sb_vert1"] - 1.07),  # left, bottom
        (vertices_highmass["left_cut"],
         vertices_highmass["psb_cut"]),  # left, top
        (vertices_highmass["sf_vert2"],
         vertices_highmass["psb_cut"]),  # right, top
        (vertices_highmass["sf_vert1"] - 0.29,
         vertices_highmass["sb_vert1"] - 1.07)  # right, bottom
    ]
    pca_patches_highmass.append([sb_verts, 'yellow'])

    # Green valley
    green_verts = [
        (vertices_highmass["sf_vert4"],
         vertices_highmass["junk_y_lower"]),  # left, bottom
        (vertices_highmass["sf_vert3"],
         vertices_highmass["psb_cut"]),  # left, top
        (vertices_highmass["green_vert1"],
         vertices_highmass["psb_cut"]),  # right, top
        (vertices_highmass["green_vert2"],
         vertices_highmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_highmass.append([green_verts, 'green'])

    # Red
    red_verts = [
        (vertices_highmass["green_vert2"],
         vertices_highmass["junk_y_lower"]),  # left, bottom
        (vertices_highmass["green_vert1"] + 4.9,
         vertices_highmass["junk_y_upper"]),  # left, top
        (vertices_highmass["right_cut"] + 0.07,
         vertices_highmass["junk_y_upper"]),  # right, top
        (vertices_highmass["right_cut"] + 0.07,
         vertices_highmass["junk_y_lower"])  # right, bottom
    ]
    pca_patches_highmass.append([red_verts, 'red'])

    # Junk
    junk_verts = [
        (vertices_highmass["sf_vert1"] - 0.3,
         vertices_highmass["junk_y_lower2"]),  # left, bottom
        (vertices_highmass["sf_vert2"] - 0.3,
         vertices_highmass["junk_y_lower"]),  # left, top
        (vertices_highmass["right_cut"] + 0.07,
         vertices_highmass["junk_y_lower"]),  # right, top
        (vertices_highmass["right_cut"] + 0.07,
         vertices_highmass["junk_y_lower2"])  # right, bottom
    ]
    pca_patches_highmass.append([junk_verts, 'grey'])

    return pca_patches_highmass

# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Robust Principal Component Analysis projection, and tools

Based on IDL implementation by Vivienne Wild

References:
  [1] Connolly & Szalay (1999, AJ, 117, 2052)
  http://www.journals.uchicago.edu/AJ/journal/issues/v117n5/980466/980466.html
  [2] Lemson, "Normalized gappy PCA projection"


Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import os
import numpy as np
import scipy.io as sciio

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
def eigenbasis(verbose=False):
    """
    Returns wavelengths, eigenspectra, mean spectrum, and variance
    for Wild+07 PCA eigensystem.


    Parameters
    ----------
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''.

    Returns
    -------
    wave : ndarray
        Wavelength array of 'float' type.
    spec : ndarray
        2D array of eiegenspectra.
    mean : ndarray
        Mean spectrum array.
    var : ndarray
        Variance array corresponding to the mean spectrum.

    """
    dir_pca = os.path.dirname(__file__).split('/')[:-1]
    fname_pca = "data/pcavo_espec_25.sav"
    path_pca = os.path.join('/'.join(dir_pca), fname_pca)
    pcavo = sciio.readsav(path_pca)

    if verbose:
        print(f"[vwpca_eigenbasis] STATUS: opened {fname_pca}")

    wave = pcavo['wave']
    spec = pcavo['espec']
    mean = pcavo['meanarr']
    var = pcavo['vararr']

    if verbose:
        print(f"[vwpca_eigenbasis] STATUS: returning eigensystem")

    return wave, spec, mean, var


def gappy(data, error, espec, mean, cov=None, \
                reconstruct=False, verbose=False):
    """
    Performs robust PCA projection.

    Parameters
    ----------
    data : ndarray
        1D spectrum or 2D specta with 'float' type.
    error : ndarray
        1D or 2D corresponding 1-sigma error array. Zeros indicate masked data.
    espec : ndarray
        2D array of eigenspectra, possibly truncated in dimension.
    mean : ndarray
        1D mean spectrum of the eigenspectra.
    cov: bool, optional
        Return covariance matrix.
        Default is ''False''.
    reconstruct : bool, optional
        Fill in missing values with PCA estimation.
        Default is ''False''.
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''

    Returns
    -------
    pcs : ndrray
        1D or 2D array of Principal Components with 'float' type.
    data : ndrray
        If reconstruct enabled, 1D or 2D reconstructed spectra.
    ccov : ndarray
        If cov enabled, 2D or 3D covariance matrices.
    """

    # Sanity checks
    if (np.size(data) == 0) | (np.size(error) == 0) | (np.size(espec) == 0) | (
            np.size(mean) == 0):
        print('[pca_gappy] ERROR: incorrect input lengths')
        return None

    tmp = np.shape(espec)  # number of eigenvectors
    if np.size(tmp) == 2:
        nrecon = tmp[0]
    else:
        nrecon = 1
    nbin = np.shape(espec)[-1]  # number of data points
    tmp = np.shape(data)  # number of observations to project
    if np.size(tmp) == 2:
        ngal = tmp[0]
    else:
        ngal = 1

    # Dimension mismatch check
    if np.shape(data)[-1] != nbin:
        print(
            '[pca_gappy] ERROR: "data" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(error)[-1] != nbin:
        print(
            '[pca_gappy] ERROR: "error" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(mean)[0] != nbin:
        print(
            '[pca_gappy] ERROR: "mean" must have the same dimension as eigenvectors'
        )
        return None

    # Project each galaxy in turn
    pcs = np.zeros((ngal, nrecon))
    if cov is not None:
        ccov = np.zeros((ngal, nrecon, nrecon))

    for j in np.arange(0, ngal):

        if ngal == 1:
            data = data[np.newaxis, :]
            error = error[np.newaxis, :]

        if verbose:
            print('[pca_gappy] STATUS: processing spectrum ')

        # Calculate weighting array from 1-sig error array
        # ! if all bins have error=0 continue to next spectrum
        weight = np.zeros(nbin)
        ind = np.where(error[j, :] != 0.)[0]
        if np.size(ind) != 0:
            weight[ind] = 1. / (error[j, :][ind]**2)
        else:
            if verbose:
                print(
                    '[pca_gappy] WARNING: error array problem in spectrum (setting pcs=0)'
                )
            continue

        ind = np.where(np.isfinite(weight) is False)[0]
        if np.size(ind) != 0:
            if verbose:
                print(
                    '[pca_gappy] WARNING: error array problem in spectrum (setting pcs=0)'
                )
            continue

        # Subtract mean from data
        data_j = data[j, :] - mean

        # Calculate the weighted eigenvectors, multiplied by the eigenvectors (eq. 4-5 [1])
        if nrecon > 1:
            # CONSERVED MEMORY NOT IMPLEMETED
            M = np.dot((espec * weight), espec.T)

            # Calculate the weighted data array, multiplied by the eigenvectors (eq. 4-5 [1])
            F = np.dot((data_j * weight), espec.T)

            # Solve for Principle Component Amplitudes (eq. 5 [1])
            try:
                Minv = np.linalg.inv(M)
            except:
                if verbose:
                    print(
                        '[pca_gappy] STATUS: problem with matrix inversion (setting pcs=0)'
                    )

                continue

            pcs[j, :] = np.dot(F, Minv)

            # Calculate covariance matrix (eq. 6 [1])
            if cov is True:
                ccov[j, :, :] = Minv

        else:  # if only one eigenvector
            M = np.sum(weight * espec * espec)
            F = np.sum(weight * data_j * espec)
            pcs[j, 0] = F / M
            if cov is True:
                ccov[j, 0, 0] = np.sum((1. / weight) * espec * espec)

        # If reconstruction of data array required,
        # fill in regions with weight = 0 with PCA reconstruction
        if reconstruct is True:
            bad_pix = np.where(weight == 0.)[0]
            count = np.size(bad_pix)
            if count == 0:
                continue

            rreconstruct = np.sum((pcs[j, :] * espec[:, bad_pix].T).T, 0)
            rreconstruct += mean[bad_pix]
            data[j, bad_pix] = rreconstruct

    if ngal == 1:
        pcs = pcs[0]
        data = data[0]
    if cov:
        ccov = ccov[0]

    # Report to user
    if verbose:
        print("[pca_gappy] STATUS: Results...")
        for i, pc in enumerate(pcs):
            print(f"               PCA{i+1}: {pc:2.5f}")

    # Return
    if reconstruct is True:
        if cov is True:
            return pcs, data, ccov
        else:
            return pcs, data

    elif cov is True:
        return pcs, ccov
    else:
        return pcs


def normgappy(data, error, espec, mean, cov=False, \
                reconstruct=False, verbose=False):
    """
    Performs robust PCA projection, including normalization estimation.

    Parameters
    ----------
    data : ndarray
        1D spectrum or 2D specta with 'float' type.
    error : ndarray
        1D or 2D corresponding 1-sigma error array. Zeros indicate masked data.
    espec : ndarray
        2D array of eigenspectra, possibly truncated in dimension.
    mean : ndarray
        1D mean spectrum of the eigenspectra.
    cov: bool, optional
        Return covariance matrix.
        Default is ''False''.
    reconstruct : bool, optional
        Fill in missing values with PCA estimation.
        Default is ''False''.
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''

    Returns
    -------
    pcs : ndrray
        1D or 2D array of Principal Components with 'float' type.
    norm: float or ndarray
        Normalization estimates.
    data : ndrray
        If reconstruct enabled, 1D or 2D reconstructed spectra.
    ccov : ndarray
        If cov enabled, 2D or 3D covariance matrices.
    """

    # Sanity checks
    if (np.size(data) == 0) | (np.size(error) == 0) | (np.size(espec) == 0) | (
            np.size(mean) == 0):
        print('[pca_normgappy] ERROR: incorrect input lengths')
        return None

    tmp = np.shape(espec)  # number of eigenvectors
    if np.size(tmp) == 2:
        nrecon = tmp[0]
    else:
        nrecon = 1
    nbin = np.shape(espec)[-1]  # number of data points
    tmp = np.shape(data)  # number of observations to project
    if np.size(tmp) == 2:
        ngal = tmp[0]
    else:
        ngal = 1

    # Dimension mismatch check
    if np.shape(data)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "data" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(error)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "error" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(mean)[0] != nbin:
        print(
            '[pca_normgappy] ERROR: "mean" must have the same dimension as eigenvectors'
        )
        return None

    # Project each galaxy in turn
    pcs = np.zeros((ngal, nrecon), float)
    norm = np.zeros(ngal, float)
    if cov is not None:
        ccov = np.zeros((ngal, nrecon, nrecon))

    if ngal == 1:
        data = data[np.newaxis, :]
        error = error[np.newaxis, :]

    for j in np.arange(0, ngal):

        if verbose:
            print('[pca_normgappy] STATUS: processing spectrum ')

        # Calculate weighting array from 1-sig error array
        # ! if all bins have error=0 continue to next spectrum
        weight = np.zeros(nbin)
        ind = error[j, :].nonzero()[0]
        if np.size(ind) != 0:
            try:
                weight[ind] = 1. / (error[j, :][ind]**2)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                    )
                continue

        ind = np.where(np.isfinite(weight) is False)[0]
        if np.size(ind) != 0:
            if verbose:
                print(
                    '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                )
            continue

        data_j = data[j, :]

        # Solve partial chi^2/partial N = 0
        Fpr = np.sum(weight * data_j * mean)  # eq 4 [2]
        Mpr = np.sum(weight * mean * mean)  # eq 5 [2]
        E = np.sum((weight * mean) * espec, axis=1)  # eq 6 [2]

        # Calculate the weighted eigenvectors, multiplied by the eigenvectors (eq. 4-5 [1])

        if nrecon > 1:
            # CONSERVED MEMORY NOT IMPLEMETED
            espec_big = np.repeat(espec[:, np.newaxis, :], nrecon, axis=1)
            M = np.sum(weight * np.transpose(espec_big, (1, 0, 2)) * espec_big, 2)

            # Calculate the weighted data array, multiplied by the eigenvectors (eq. 4-5 [1])
            F = np.dot((data_j * weight), espec.T)

            # Calculate new M matrix, this time accounting for the unknown normalization (eq. 11 [2])
            E_big = np.repeat(E[np.newaxis, :], nrecon, axis=0)
            F_big = np.repeat(F[:, np.newaxis], nrecon, axis=1)
            Mnew = Fpr * M - E_big * F_big

            # Calculate the new F matrix, accounting for unknown normalization
            Fnew = Mpr * F - Fpr * E

            # Solve for Principle Component Amplitudes (eq. 5 [1])
            try:
                Minv = np.linalg.inv(Mnew)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] STATUS: problem with matrix inversion (setting pcs=0)'
                    )

                continue

            pcs[j, :] = np.squeeze(np.sum(Fnew * Minv, 1))
            norm[j] = Fpr / (Mpr + np.sum(pcs[j, :] * E))

            # Calculate covariance matrix (eq. 6 [1])
            if cov is True:
                M_gappy = np.dot((espec * (weight * norm[j]**2)), espec.T)
                ccov[j, :, :] = np.linalg.inv(M_gappy)

        else:  # if only one eigenvector
            M = np.sum(weight * espec * espec)
            F = np.sum(weight * data_j * espec)
            Mnew = M * Fpr - E * F
            Fnew = Mpr * F - E * Fpr
            pcs[j, 0] = Fnew / Mnew
            norm[j] = Fpr / (Mpr + pcs[j, 0] * E)
            if cov is True:
                ccov[j, 0, 0] = np.sum((1. / weight) * espec * espec)

        # If reconstruction of data array required,
        #   fill in regions with weight = 0 with PCA reconstruction
        if reconstruct is True:
            bad_pix = np.where(weight == 0.)
            count = np.size(bad_pix)
            if count == 0:
                continue

            rreconstruct = np.sum((pcs[j, :] * espec[:, bad_pix].T).T, 0)
            rreconstruct += mean[bad_pix]
            data[j, bad_pix] = reconstruct

    if ngal == 1:
        pcs = pcs[0]
        data = data[0]
        norm = norm[0]
        if cov:
            ccov = ccov[0]

    # Report to user
    if verbose:
        print("[pca_normgappy] STATUS: Results...")
        for i, pc in enumerate(pcs):
            print(f"               PCA{i+1}: {pc:2.5f}")
        print(f"               Norm: {norm:2.5f}")

    # Return
    if reconstruct is True:
        if cov is True:
            return pcs, norm, data, ccov
        else:
            return pcs, norm, data

    elif cov is True:
        return pcs, norm, ccov
    else:
        return pcs, norm

def mc_errors(data, error, espec, emean, Ntrials=100, verbose=False):
    """
    Performs Monte-Carlo error estimation assuming a normal distribution.
    The formal 1-sigma errors from the (norm)gappy cov matrix should match.

    Parameters
    ----------
    data : ndarray
        1D spectrum or 2D specta with 'float' type.
    error : ndarray
        1D or 2D corresponding 1-sigma error array. Zeros indicate masked data.
    espec : ndarray
        2D array of eigenspectra, possibly truncated in dimension.
    mean : ndarray
        1D mean spectrum of the eigenspectra.
    Ntrials: int, optional
        Number of trials to perform
        Default is 100.
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''

    Returns
    -------
    pc_errors : ndrray
        1D array of Principal Component errors with 'float' type.
    """

    mcdata = np.random.normal(data, error, size=(Ntrials, len(data)))
    mcerror = error[None,:].repeat(Ntrials, 0)
    mcpcs = gappy(
        mcdata, mcerror, espec, emean, cov=False, verbose=verbose)
    pc_errors = np.std(mcpcs, 0)

    return pc_errors

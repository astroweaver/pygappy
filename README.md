pygappy
==============================

Robust PCA Projection

About
-----

Written by John Weaver (St Andrews/Copenhagen 2018)

Based on the original Principal Component Analysis (PCA) projection routine
by Vivienne Wild, including fitting around bad values (gappy), and unknown
normalization factor (norm-gappy). For information about gappy PCA, please
refer to Wild et al. 2007.

You can test the installation and/or try-out the software by running the
"pca_tryme.py" script.


References
----------
[1] Connolly & Szalay (1999, AJ, 117, 2052)
http://www.journals.uchicago.edu/AJ/journal/issues/v117n5/980466/980466.html
[2] Lemson, "Normalized gappy PCA projection"


Quick How-to
------------

1. If required, set the error value of any excluded points to zero.

2. (optional) Apply a Galactic dust correction of your choice.

3. If not done already, shift wavelength array to restframe. Interpolation
   should be avoided where possible.

4. Select N number of eigenspectra to apply. If N is less than 3 (not advised),
   also pass that number to visualization.pca_plot([args], Nshow = N).

5. Choose either pca_gappy or pca_normgappy to project eigenbasis onto data
   array. Note: The mean spectrum argument should always take the mean array
   of the original eigenbasis!

6. Extract 1-sigma errors by setting cov = True and run np.diag(cov).


Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── config
    ├── data
    │   ├── pcavo_espec_25.sav
    ├── docs
    ├── pca_tryme.py
    └── src
        ├── __init__.py
        ├── pca_projection.py
        ├── visualisation.py

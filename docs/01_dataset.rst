********
Data Set
********


Structure
=========

Conceptional Structure
----------------------
The data provides a mapping from multiple *NOAA active region* images,
perceived by SDO, to a peak flux which has to be predicted.
The active region images are provided over a duration of 12 hours in a 1
hour cadence. The peak flux prediction window are the next 24 hours after the
last input.

The data set is split into *training* and *test* samples.
A single sample consists of 12 *time steps*, being each 1 hour apart,
and a target peak flux.

A time step consists of multiple 512x512 image patches.
Each image patch corresponds to a single AIA wavelength.
If images for all wavelengths are available, a single time step thus contains
10 image patches, less otherwise. A single time step is a compressed Numpy
file where the array keys stored inside correspond to the wavelength they were
perceived from.

The training and test sets are sampled in a way which prevents training
set bias to influence test set results. Active regions in the training and
test sets are mutually exclusive.

The data set directory structure looks as follows::

    - root
        - training
            - ...
        - test
            - {sample id 1}
                - {time step 1}.npz
                - {time step 2}.npz
                - ...
            - {sample id 2}
                - ...

The format for the time step files is (in Python date time format syntax)::

    %Y-%m-%dT%H%M%S.npz


Parameters
==========

.. todo::
    Document data set parameters like duration or cadence.


Creation Algorithm
==================

.. todo::
    Data set creation algorithm.


Open Points
===========

.. todo::
    Document the points which are still open.

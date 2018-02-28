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
last input. The input images are taken directly from FITS files without any
data normalization or gamma correction. The flare peak fluxes stem from
*SWPC* data.

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
Various points are still open due to time constraints.

Conceptional
------------
- A single active region can split into multiple new active regions and
  multiple active regions can merge into a single one.
  It has to be checked how such events manifest in HEK events to make sure
  no accidental bias between test and training sets is introduced.
- Due to merging and splitting, but also due to bad data, some active region
  events might overlap each other spatially. Some verification is needed to be
  sure no two active regions of the test and training set overlap each other,
  otherwise parts of image patches are present in both sets.
- It might be that a NOAA active region produces a flare which is not archived
  by SWPC. Non-flaring samples have to be verified to make sure no wrong
  output peak flux is provided.
- The SWPC flare to NOAA number matching partially relies on
  *SSW Latest Events* data. It was not determined yet if those events are
  reliable.
- The peak flux for non-flaring active region has to be provided in some form.
  Fluxes in the *GOES* light curve are not reliable as they capture the fluxes
  from **all** of the sun's active regions. The region peak flux has to be
  either calculated in some way or approximated using a constant, low value.
- A set of image header values is currently checked to see if instrument issues
  or an earth eclipse is visible on the target image. The checks used should
  be verified and it has to be checked if a more reliable method exists.

Implementation
--------------
- HMI data should also be provided as an input. This has not been done yet.
- SDO sensors collect less data over time
  (see https://github.com/Helioviewer-Project/helioviewer.org/issues/136).
  It has to be decided if this is left like that intentionally or if some form
  of intensity adjustment should be performed.
- At the moment, more meta-data columns are written than necessary.
  The amount of output has to be reduced to a sensible level.
- Image intensities are currently saved as ``float64`` values, after being
  cast from ``int16`` values by the *SunPy* library. ``float64`` values take
  up a large amount of space and cannot easily be compressed.
  However, a larger data type than ``int16`` is necessary as intensities can
  become larger during processing. Either ``int32`` or ``uint16`` values
  should be used. The disadvantage of ``uint16`` is that negative values are
  clipped, thus changing the measurements. On the other hand, ``int32`` values
  might result in a data set size which is difficult to handle.

General
-------
- The sampling (especially the selection of input time ranges) might currently
  not be stochastically correct and needs to be verified.

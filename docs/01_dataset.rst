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

The data set creation is divided into the following steps:

1. Data loading
2. Event processing
3. Sampling
4. Sampling validation
5. Output creation

All steps use existing data if already present and only perform
actions for missing data.

Data Loading
------------
Initially, the raw event list for all flares and active regions
in the data set time frame is downloaded from the
`Heliophysics Events Knowledgebase (HEK) <https://www.lmsal.com/hek/>`_.
The events are stored in a JSON file for later usage.

Aditionally, the GOES x-ray flux for the data set time frame is
downloaded from the `NOAA archive <https://satdat.ngdc.noaa.gov/sem/>`_
and stored in a separate directory, resulting in a file for each day.

Event Processing
----------------
During the event processing, each NOAA active region is split into *ranges*.
A *range* consists of a start time, end time, and the GOES class of the largest
flare occurring in that range (if any). Each range is at least of prediction
period length. Thus, a range is a part of an active region during which
the same prediction is expected.

Initially, SWPC flares and NOAA active region events are extracted from the
raw HEK event list. Multiple HEK events belonging to the same NOAA active region
are grouped together. Each SWPC flare is then mapped to a NOAA active region.
The mapping is later used to determine cut-out coordinates as SWPC events often
lack coordinate values and to slice the active region's duration into ranges.

The mapping is performed as follows:

1. If the flare has a NOAA number assigned, it is mapped to that active region.
2. Otherwise, the closest *SSW Latest Events* flare event is searched by comparing
   the peak, end and start time distance.
3. If the peak time delta is not larger than 1 minute or the start and end times
   of both events match and the peak time delta is not larger than 10 minutes,
   the SSW event is considered to be equal to the SWPC event.
4. If the SSW event is considered equal and has a NOAA number assigned, the SWPC
   event is mapped to the corresponding NOAA active region.

This results in a list of SWPC flare events for each NOAA active region and an
additional list of SWPC flares which could not be assigned to any active region.

Each active region is individually considered and split into ranges.

First, the whole active region duration is considered *free* (usable as non-flaring
prediction). Afterwards, each flare mapped to the active region is individually processed.

The range in which a prediction window would contain the flare is chopped from the free ranges.
This ensures that wherever inside a free range a prediction window is placed,
the prediction window will never contain any flare.

A flare's range has to fulfill the following criterias:

- The prediction window can be placed anywhere inside the range while containing the flare.
- No other flare's peak flux is larger than the one of the flare currently observed.
- The distance between the range start and the active region start is at lest as long as the
  input duration, guaranteeing that whenever the prediction window starts, all inputs are
  in the active region's time range.

After all flares have been processed, the active region has a list of ranges which are either
flaring or free and non-intersecting.

The ranges are post-processed by chopping out the durations of all SWPC flares which could not
be assigned to any NOAA active region. This way, the prediction target for each range is
guaranteed to not be accidentally too low.

Finally, all ranges which are shorter than the prediction period are discarded as they are
of no use.

Sampling
--------
During the sampling step, NOAA active regions are first split into test and training sets
and afterwards processed to create actual samples for the active region ranges.

To ensure an unbiased test set, each active region is assigned to only one set.
First, active regions are grouped by their largest flare's goes class (letter and first digit).
Active regions without any flares are grouped into a separate *free* group.

Test set active regions are then sampled from those groups (except *free*)
by looking at each group individually:

- If the group contains less than 6 active regions, a single random active region is
  assigned to the test set with a 50% chance.
- Otherwise, 3 active regions are assigned to the test set at random.

Afterwards, active regions from the *free* group are assigned to the test set at random.
The number of active regions to use is 1/4th of the number of flaring active regions in
the test set. This way, the number of free active regions in the test set is roughly equal
to the number of active regions per general GOES class (B, C, M and X).

All active regions which were not assigned to the test set are then assigned to the training set.

Individual active regions in each set are further processed to create actual samples.
Each active region range is split into a number of samples, each sample being an input time window
and a target prediction. Input time windows are not allowed to overlap, thus creating an upper bound
of the number of samples resulting from a single range.
The minimum number of samples of a range is determined as follows:

- If the range's target prediction is an M or larger flare and the maximum number of samples
  is more than 1, the minimum number of samples is 2.
- Otherwise, the minimum number of samples is 1.

The number of samples is then uniformly chosen between the minimum and maximum number of samples.
The chosen number of input windows are then randomly taken from the range so that no two input windows
overlap.

It has to be noted that a active region range defines a prediction period. Thus, the first possible
input window starts before the region range and the last possible input window ends before the
range end.

Sampling Validation
-----------------
Created samples are validated to catch conceptual or implementation issues.

First, it is ensured that no active region is present in both the test and training set.
Afterwards, each sample is validated individually by checking the following:

- Is the duration of each sample equal to the input duration?
- Does each sample's peak flux happen after the input duration?
- Does each sample's peak flux happen during in the prediction window?
- Is each sample's input duration fully contained in its active region duration?
- Is each sample's prediction window fully contained in its active region duration?

If any validation fails, no output is created.

Output Creation
---------------

.. todo::
    Document output creation.


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

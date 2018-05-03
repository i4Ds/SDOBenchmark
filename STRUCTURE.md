Structure
=========

Conceptional Structure
----------------------
This dataset provides multiple samples of rendered SDO AIA and HMI FITS images of *NOAA active regions*, alongside a GOES X-ray peak flux which has to be predicted. The active region images are created at 4 time steps before the peak flux prediction window (12h, 5h, 1.5h and 10min). The peak flux prediction windows are the next 24 hours. The flare peak fluxes stem from *SWPC* data.

The data set is split into *training* and *test* samples. A single sample consists of 4 *time steps*, being each 1 hour apart, and a target peak flux.

A time step consists of multiple 512x512 image patches. Each image patch corresponds to a single AIA wavelength or HMI image. If images for all wavelengths are available, a single time step thus contains 10 image patches, less otherwise. All images of a sample are within one folder, therefore containing up to 40 images.

The training and test sets are sampled in a way which prevents training set bias to influence test set results: Active regions in the training and test sets are mutually exclusive.



Parameters
==========
Various parameters change the created data set.
The following table lists all parameters and their defaults:

| Parameter     | Description                           | Default    |
| ------------- | ------------------------------------- | ---------- |
| start         | First time (inclusive) of the period for which samples are created | 01.01.2012 |
| end           | Last time (exclusive) of the period for which samples are created | 01.01.2018 |
| input_hours   | Number of hours for which input is provided before a prediction has to be made | 12 hours   |
| output_hours  | Number of hours after the last input for which a prediction has to be made | 24 hours   |
| time_steps | Input image time stamps after input start (i.e. starting 12h before prediction window) | [0, 7*60, 10*60+30, 11*60+50] minutes     |
| seed          | Seed used to initialize the random number generator for sampling | 726527     |


Creation Algorithm
==================

The data set creation is divided into the following steps:

1. Data loading
2. Event processing
3. Sampling
4. Sampling validation
5. Output creation

All steps use existing data if already present and only perform actions for missing data.

Data Loading
------------
Initially, the raw event list for all flares and active regions in the data set time frame is downloaded from the [Heliophysics Events Knowledgebase (HEK)](https://www.lmsal.com/hek/). The events are stored in a JSON file for later usage.

Aditionally, the GOES x-ray flux for the data set time frame is downloaded from the [NOAA archive](https://satdat.ngdc.noaa.gov/sem/) and stored in a separate directory, resulting in a file for each day.

Event Processing
----------------
During the event processing, each NOAA active region is split into *ranges*. A *range* consists of a start time, end time, and the GOES class of the largest flare occurring in that range (if any). Each range is at least of prediction period length (24h per default). Thus, a range is a part of an active region during which the same prediction is expected.

Initially, SWPC flares and NOAA active region events are extracted from the raw HEK event list. Multiple HEK events belonging to the same NOAA active region are grouped together. Each SWPC flare is then mapped to a NOAA active region. The mapping is later used to determine cut-out coordinates, as SWPC events often lack coordinate values, and to slice the active region's duration into ranges.

The mapping is performed as follows:

1. If the flare has a NOAA number assigned, it is mapped to that active region.
2. Otherwise, the closest *SSW Latest Events* flare event (goes flare list + location) is searched by comparing the peak, end and start time distance.
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

A flare's range has to fulfill the following criteria:

- The prediction window can be placed anywhere inside the range while containing the flare.
- The flare's peak flux is highest in its range.
- The distance between the range start and the active region start is at least as long as the  input duration (default 12h), guaranteeing that whenever the prediction window starts, all inputs are in the active region's time range.

Parts of the free range are removed where the GOES curve exceeds 8e-9.

After all flares have been processed, the active region has a list of ranges which are either
flaring or free and which are non-intersecting.

The ranges are post-processed by chopping out the durations of all SWPC flares which could not
be assigned to any NOAA active region. This way, the prediction target for each range is
guaranteed to not be accidentally too low.

Finally, all ranges which are shorter than the prediction period are discarded as they are
of no use.

Sampling
--------
During the sampling step, NOAA active regions are first split into test and training sets
and afterwards processed to create actual samples for the active region ranges.

To ensure an unbiased test set, each active region is assigned to only one set. First, active regions are grouped by their largest flare's GOES class (letter and first digit). Active regions without any flares are grouped into a separate *free* group.

Test set active regions are then sampled from those groups (except *free*) by looking at each group individually:

- If the group contains less than 6 active regions, a single random active region is assigned to the test set with a 50% chance.
- Otherwise, 3 active regions are assigned to the test set at random.

The X flare of September 2017 is an exception and will always be in the test set.

Afterwards, active regions from the *free* group are assigned to the test set at random. We put 1/4th of the number of flaring active regions in the test set.

All active regions which were not assigned to the test set are then assigned to the training set.

Individual active regions in each set are further processed to create actual samples.
Each active region range is split into a number of samples, each sample being an input time window and a target prediction. Input time windows are not allowed to overlap, thus creating an upper bound of the number of samples resulting from a single range. The minimum number of samples of a range is determined as follows:

- If the range's target prediction is an M or larger flare and the maximum number of samples
  is more than 1, the minimum number of samples is 2.
- Otherwise, the minimum number of samples is 1.

The number of samples is then uniformly chosen between the minimum and maximum number of samples (because we do not want neural nets to base predictions on sample interval times).
The chosen number of input windows are then randomly taken from the range so that no two input windows overlap.

It has to be noted that an active region range defines a prediction period. Thus, the first possible input window starts before the region range and the last possible input window ends before the range end.

Sampling Validation
-------------------
Created samples are validated to catch conceptual or implementation issues.

First, it is ensured that no active region is present in both the test and training set.
Afterwards, each sample is validated individually by checking the following:

- Is the duration of each sample equal to the input duration?
- Does each sample's peak flux happen after the input duration?
- Does each sample's peak flux happen during in the prediction window?
- Is each sample's input duration fully contained in its active region duration?
- Is each sample's prediction window fully contained in its active region duration?
  (although this is not necessary for predictions at the limb, it is an easy way to prevent 
  wrong overlaps when e.g. an active region's number changes)

If any validation fails, no output is created.

Output Creation
---------------
Finally, the actual samples are created in three steps:

1. FITS data over the input duration is requested from JSOC.
2. The FITS "images" of a completed request are downloaded.
3. Downloaded FITS files are processed to create output images.

Due to the nature of the data, the output creation is parallelized. Each of the three steps are executed in parallel for a number of samples at the same time.

The creation of samples is working with plenty of retries and fallbacks, e.g. retrying after connection issues or extending FITS files with additionally requested header keys.

JSOC requests are issued in the *as-is* format and *url-quick* protocol. Consult [drms.readthedocs.io](http://drms.readthedocs.io/en/stable/tutorial.html#url-quick-as-is) for further details.

FITS files are downloaded into a temporary *_fits_temp* directory. This directory will be deleted after the downloaded images have been processed. A single downloaded FITS file represents a single wavelength at a single time, in AIA level 1.0 format.

After all FITS files of a sample are downloaded, they are further processed. First, because some files can be missing, the downloaded FITS files have to be assigned to individual time steps in the input cadence. For each time step, each file for each frequency is processed as follows:

1. FITS header values are verified to check if instrument or other issues
   (e.g. an earth eclipse) are present on the image.
   If yes, the image is discarded.
2. AIA level 1 to level 1.5 processing is performed.
3. The target active region coordinates with regard to solar rotation
   and the time difference is calculated on the current image.
4. A patch around the rotated coordinates is cut out
5. The resulting data is clipped by predefined clipping ranges
6. Then the image is saved to disk with half resolution as 8-bit JPEG.

Open Points
===========
Various points are still open due to time constraints.

Conceptional
------------
- A single active region can split into multiple new active regions and multiple active regions can merge into a single one. It has to be checked how such events manifest in HEK events to make sure no accidental bias between test and training sets is introduced.
- Due to merging and splitting, but also due to bad data, some active region events might overlap each other spatially. Some verification is needed to be sure no two active regions of the test and training set overlap each other, otherwise parts of image patches are present in both sets.
- A set of FITS header values is currently checked to see if instrument issues or an earth eclipse is visible on the target image. The checks used should be verified and it has to be checked if other methods exist.



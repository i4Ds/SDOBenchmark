Noisy images
============

Some raw data from the satellite is flagged as noisy, e.g. because of a moon eclipse or because of a recalibration. The images were still created, yet they have a string "flagged" in the ImageDescription metadata (EXIF).


File Structure
==============

The folder layout is irrelevant for training, because each sample's id in 'meta_data.csv' maps to a sample.
See e.g. https://github.com/i4ds/SDOBenchmark/blob/master/notebooks/utils/keras_generator.py#L61
Except if you want to split the training set further, e.g. for cross validation: Make the splitting on the first layer of folders ('11402', '13386', ...).

If you still want to know more:
The folders "test" and "training" contain folders with numbers like '11402' or '13386'. Those numbers are Active Region numbers, i.e. enumarations for patches of the sun where we take samples from.
Within those folders are the sample folders. Each sample folder's name is a combination of a date and a number, e.g. '2014_01_28_12_50_00_1'. The date defines when a range starts, the last digit is the sample number. A range is a time period during which the maximum flux doesn't exceed its label 'peak_flux'.

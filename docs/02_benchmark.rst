*********
Benchmark
*********

The benchmark defines two ways to measure a model's performance:
an absolute perfomance metric and a set of relative performance metrics.

The absolute performance metric is a single number which can be
used to compare different models and determine which performs best.

The relative performance metrics are useful to determine which parts
of a single model need improvement.


Absolute Performance Metric
===========================
The absolute performance metric is the *Root Mean Squared Logarithmic Error*
between predicted and true peak fluxes for five groups of samples.
The mean of the five individual metrics produces a single comparable number.

The *Root Mean Squared Logarithmic Error* between the ground truth :math:`y` and
predictions :math:`\hat{y}` for :math:`n` samples is defined as follows:

.. math::

    \sqrt{\frac{1}{n} \sum_{i=1}^n{\left(\log(y_i) - \log(\hat{y_i})\right)^2}}

The logarithm with base 10 is used because it directly correlates with the
GOES class classification scheme.

Samples are grouped by their true peak flux, converted to a GOES class.
All B, C, M, and X classified peaks are each grouped together.
The fivth group consists of all *non-flaring* or *free* samples respectively.
This way, the bias present in the peak flux distribution is not present
in the metric.


Relative Performance Metrics
============================
Relative performance metrics aid in the discovery of strengths and
weaknesses of a particular model.

All samples are grouped into five groups as done for the absolute
performance metric. By treating those groups as *classes*,
the following classification metrics can be calculated per class:

- Accuracy
- Precision
- Recall

Additionally, a confusion matrix over the five classes can be
calculated.

Those groups can further be combined to result in only two groups,
one containing all M and larger flares, the other containing
all smaller flares as well as non-flaring samples.
The same classification metrics as mentioned above can then
be calculated. The resulting metrics show how well a model
works as an early warning system due to the detection of
possibly significant flares.

.. todo::
    It would also be interesting to use the binary classification
    evaluation to create a ROC curve.

    The curve could be created by varying the threshold which separates
    *significant* from *insignificant* peak fluxes.


Open Points
===========
Various points are still open due to time constraints.

Implementation
--------------
- The benchmark has to be implemented. One possibility is to use two CSV files
  as an input (one for the ground truth, one for the model predictions).
  Each CSV file contains two columns: the sample id and its peak flux.

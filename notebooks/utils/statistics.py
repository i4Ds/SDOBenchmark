import numpy as np
import math
from sklearn.metrics import confusion_matrix

goes_classes = ['quiet','A','B','C','M','X']


def flux_to_class(f: float, only_main=False):
    'maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification'
    decade = int(min(math.floor(math.log10(f)), -4))
    main_class = goes_classes[decade + 9] if decade >= -8 else 'quiet'
    sub_class = str(round(10 ** -decade * f)) if main_class != 'quiet' and only_main != True else ''
    return main_class + sub_class

def class_to_flux(c: str):
    'Inverse of flux_to_class \
    Maps a flare class (e.g. B6, M, X9) to a GOES flux value'
    if c == 'quiet':
        return 1e-9
    decade = goes_classes.index(c[0])-9
    sub = float(c[1:]) if len(c) > 1 else 1
    return round(10 ** decade * sub, 10)


#
#   See https://arxiv.org/pdf/1608.06319.pdf for details about scores and statistics
#

def true_skill_statistic(y_true, y_pred, threshold='M'):
    'Calculates the True Skill Statistic (TSS) on binarized predictions\
    It is not sensitive to the balance of the samples\
    This statistic is often used in weather forecasts (including solar weather)\
    1 = perfect prediction, 0 = random prediction, -1 = worst prediction'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) - fp / (fp + tn)

def heidke_skill_score(y_true, y_pred, threshold='M'):
    'Claculates the Heidke skill score'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp + fn) / len(y_pred) * (tp + fp) / len(y_pred) + (tn + fn) / len(y_pred) * (tn + fp) / len(y_pred)

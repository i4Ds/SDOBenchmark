import collections
import datetime as dt
import logging
from typing import List, Tuple, Dict

import intervaltree
import numpy as np
import pandas as pd
import os
import csv
import math

from multiprocessing import Pool

import dataset.util as util

logger = logging.getLogger(__name__)


def extract_events(raw_events: List[dict]) -> Tuple[List[dict], Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]]]:

    noaa_regions = dict()
    for event in raw_events:
        event['starttime'] = util.hek_date(event["event_starttime"])
        event['endtime'] = util.hek_date(event["event_endtime"])
        if event["event_type"] == "FL":
            event['peaktime'] = util.hek_date(event["event_peaktime"])
            event['fl_peakflux'] = class_to_flux(event["fl_goescls"]) / 0.85

        # Extract NOAA active region events and group them by their number
        if event["event_type"] == "AR" and event["frm_name"] == "NOAA SWPC Observer":
            noaa_id = event["ar_noaanum"]

            if noaa_id not in noaa_regions:
                noaa_regions[noaa_id] = [event]
            else:
                noaa_regions[noaa_id].append(event)

    # Sort active regions by start and end times
    noaa_regions = {
        noaa_id: (
            min(map(lambda event: event["starttime"], events)),
            max(map(lambda event: event["endtime"], events)),
            _clean_duplicate_noaa_events(events)
        )
        for noaa_id, events in noaa_regions.items()
    }

    # Extract SWPC dataset and sort by start, peak and end times
    swpc_flares = list(sorted(
        (event for event in raw_events if event["event_type"] == "FL" and event["frm_name"] == "SWPC"),
        key=lambda event: (
            event["starttime"],
            event["peaktime"],
            event["endtime"]
        )
    ))

    return swpc_flares, noaa_regions


def _clean_duplicate_noaa_events(events: List[dict]) -> List[dict]:
    # Multiple events for the same duration can be present, in that case, use the most recent one
    date_grouping = collections.defaultdict(list)
    for event in events:
        date_grouping[event["event_starttime"]].append(event)

    return [
        group[0] if len(group) == 1 else max(group, key=lambda event: util.hek_date(event["frm_daterun"]))
        for group in date_grouping.values()
    ]


def map_flares(
        swpc_flares: List[dict],
        noaa_regions: Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]],
        raw_events: List[dict]
) -> Tuple[List[Tuple[dict, int]], List[dict]]:
    # First, map all SWPC dataset which contain a NOAA number
    mapped_flares = []
    unmapped_flares = []
    for event in swpc_flares:
        region_number = event["ar_noaanum"]

        if region_number > 0:
            if region_number in noaa_regions.keys():
                mapped_flares.append((event, region_number))
            else:
                logger.warning("NOAA active region %d referenced but not in data set, will be ignored", region_number)
                unmapped_flares.append(event)
        else:
            assert region_number == 0
            unmapped_flares.append(event)

    logger.debug("Mapped %d SWPC dataset which contained a valid NOAA active region number", len(mapped_flares))
    logger.debug("Could not map %d SWPC dataset in first step", len(unmapped_flares))

    # Try to match all unmapped dataset to SSW events and use their NOAA number if present
    ssw_flares = filter(
        lambda event: event["event_type"] == "FL" and event["frm_name"] == "SSW Latest Events", raw_events
    )
    ssw_flares = [
        (
            event["starttime"],
            event["peaktime"],
            event["endtime"],
            event
        )
        for event in ssw_flares
    ]

    still_unmapped_flares = []
    for event in unmapped_flares:
        # Find matching SSW event by peak time
        match_start, match_peak, match_end, match_event = min(
            ssw_flares,
            key=lambda candidate_event: (
                abs(event["peaktime"] - candidate_event[1]),
                abs(event["endtime"] - candidate_event[2]),
                abs(event["starttime"] - candidate_event[0])
            )
        )

        # Check peak time delta
        if abs(event["peaktime"] - match_peak) > dt.timedelta(minutes=1):
            # Allow for 10 minute peak delta if start and end times are equal
            if event["starttime"] != match_start \
                    or event["endtime"] != match_end \
                    or abs(event["peaktime"] - match_peak) > dt.timedelta(minutes=10):
                still_unmapped_flares.append(event)
                continue

        # Check if event peak is inside match duration
        if not (match_start <= event["peaktime"] <= match_end):
            still_unmapped_flares.append(event)
            continue

        # Times are equal, check if NOAA number is found
        region_number = match_event["ar_noaanum"]

        if region_number == 0:
            still_unmapped_flares.append(event)
            continue

        if event["fl_goescls"] != match_event["fl_goescls"]:
            still_unmapped_flares.append(event)
            continue

        # SSW stores NOAA numbers as 4 digits, add the missing digit
        assert region_number < 10000
        region_number += 10000
        if region_number in noaa_regions.keys():
            mapped_flares.append((event, region_number))
        else:
            logger.warning("NOAA active region %d referenced but not in data set, will be ignored", region_number)
            still_unmapped_flares.append(event)

    logger.debug("Mapped %d SWPC dataset after second step", len(mapped_flares))
    logger.debug("Could not map %d SWPC dataset after second step", len(still_unmapped_flares))

    return mapped_flares, still_unmapped_flares


def active_region_time_ranges(
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        noaa_regions: Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]],
        flare_mapping: List[Tuple[dict, int]],
        unmapped_flares: List[dict],
        goes: pd.DataFrame,
        goes_directory: str
) -> Dict[int, intervaltree.IntervalTree]:

    # Create interval tree of unmapped ranges for fast lookup later on
    global unmapped_ranges
    unmapped_ranges = intervaltree.IntervalTree()
    for event in unmapped_flares:
        unmapped_start = event["starttime"]
        unmapped_end = event["endtime"]
        unmapped_ranges.addi(unmapped_start, unmapped_end)
    unmapped_ranges.merge_overlaps()

    # Create interval tree of mapped dataset on the entire sun for fast lookup
    mapped_flares_entire_sun = intervaltree.IntervalTree()
    for flare_event, _ in flare_mapping:
        mapped_flares_entire_sun.addi(flare_event["starttime"], flare_event["endtime"])
    mapped_flares_entire_sun.merge_overlaps()

    # Construct interval where GOES > 8e-9
    goes_interval_tuples = []
    goes_interval_tuple_path = os.path.join(goes_directory, 'goes_interval_tuples.csv')
    if not os.path.exists(goes_interval_tuple_path):
        logger.info("GOES interval tuples not found, will be created at %s", goes_interval_tuple_path)
        interval_start_date = None
        prev_flux = 0
        prev_date = None
        for date, flux in goes.itertuples():
            if max(flux, prev_flux) > 8e-9: # some outlier stability
                if interval_start_date is None:
                    interval_start_date = date
            else:
                if interval_start_date is not None:
                    if date - interval_start_date > dt.timedelta(minutes=5):
                        # only add a GOES interval if it's at least 5 minutes long
                        goes_interval_tuples.append((interval_start_date, prev_date))
                    interval_start_date = None
            prev_flux = flux
            prev_date = date
        with open(goes_interval_tuple_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            for t in goes_interval_tuples:
                csvwriter.writerow(t)
    else:
        with open(goes_interval_tuple_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=';')
            for interval_start_date, prev_date in readCSV:
                goes_interval_tuples.append((
                    dt.datetime.strptime(interval_start_date.split('.')[0], '%Y-%m-%d %H:%M:%S'),
                    dt.datetime.strptime(prev_date.split('.')[0], '%Y-%m-%d %H:%M:%S')
                ))
    goes_interval = intervaltree.IntervalTree.from_tuples(goes_interval_tuples)
    del goes_interval_tuples

    # tree where GOES > 8e-9 but no (mapped) dataset present
    global goes_noflare
    goes_noflare = goes_interval - mapped_flares_entire_sun

    # mp test
    global goes_global
    goes_global = goes
    global flare_mapping_global
    flare_mapping_global = flare_mapping
    global ar_range_params
    ar_range_params = [input_duration, output_duration]

    ar_pool = Pool(10)
    ar_pool_res = ar_pool.map(_process_ar_range, noaa_regions.items())
    print('noaa regions mapped!')

    result = dict()
    for noaa_id, updated_ranges in ar_pool_res:
        result[noaa_id] = updated_ranges

    logger.warning("Approximating peak flux for of free ranges as 1e-9")

    return result

def _process_ar_range(ar_ev):
    noaa_id, (region_start, region_end, region_events) = ar_ev
    input_duration, output_duration = ar_range_params

    flares = list(sorted(
        (flare_event for flare_event, flare_region_id in flare_mapping_global if flare_region_id == noaa_id),
        key=lambda flare_event: flare_event["peaktime"]
    ))

    # Create initial free range (add 1 second to end because region_end is inclusive but intervaltree is exclusive)
    free_ranges = intervaltree.IntervalTree()
    free_ranges.addi(region_start + input_duration, region_end + dt.timedelta(seconds=1), ("free", None))

    # Process each flare
    flare_ranges = intervaltree.IntervalTree()

    flare_goes_data = [None] * len(flares)

    for idx, flare_event in enumerate(flares):
        flare_peak = flare_event["peaktime"]
        flare_class = flare_event["fl_goescls"]

        # Calculate best case range
        # 1 second deltas are used to make sure values are correctly inclusive/exclusive in ranges
        range_min = max(flare_peak - output_duration + dt.timedelta(seconds=1), region_start + input_duration)
        range_max = min(flare_peak + output_duration, region_end + dt.timedelta(seconds=1))
        if range_max - range_min >= output_duration:
            event_range = intervaltree.IntervalTree()
            event_range.addi(range_min, range_max, (flare_class, flare_peak))

            if flare_goes_data[idx] is None:
                flare_goes_data[idx] = goes_global[(goes_global.index >= flare_event["starttime"]) &
                                                    (goes_global.index <= flare_event["endtime"])]

            if len(flare_goes_data[idx]) > 0:
                flare_event['goes_peakflux'] = max(flare_goes_data[idx]['A_FLUX'])
                flare_event['goes_peaktime'] = flare_goes_data[idx]['A_FLUX'].idxmax()
            else:
                flare_event['goes_peakflux'] = class_to_flux(flare_event['fl_goescls']) / 0.85
                flare_event['goes_peaktime'] = flare_event['peaktime']

            # iff there's a bigger flare around from the same AR (HEK),
            # check with GOES curve for correct cuts.
            for idx2, flare_event2 in enumerate(flares):
                if flare_event2['fl_peakflux'] > flare_event['fl_peakflux'] * 1.1:
                    if flare_goes_data[idx2] is None:
                        flare_goes_data[idx2] = goes_global[(goes_global.index >= flare_event2["starttime"]) &
                        (goes_global.index <= flare_event2["endtime"])]
                    higher_range = flare_goes_data[idx2]


                    if len(higher_range) <= 0:
                        logger.warning("No GOES values found for flare")

                    # rewrite with numpy indexing
                    a = np.array(np.logical_or(higher_range['A_FLUX'] > flare_event['goes_peakflux'],
                         higher_range['A_FLUX'].shift(1) > flare_event['goes_peakflux']))

                    if len(a) <= 0:
                        continue

                    block_starts = np.where(np.logical_and(a[1:], np.invert(a[:-1])))[0]
                    block_starts += 1
                    if a[0] == True:
                        block_starts = np.insert(block_starts, 0, 0)

                    block_ends = np.where(np.logical_and(np.invert(a[1:]), a[:-1]))[0]
                    if a[-1] == True:
                        block_ends = np.append(block_ends, len(a) - 1)

                    for i in range(len(block_starts)):
                        if block_starts[i] != block_ends[i]: # no need to chop out a range of 0
                            event_range.chop(higher_range.index[block_starts[i]],higher_range.index[block_ends[i]] + dt.timedelta(seconds=1))


            # check if there are any ranges left that satisfy our criteria
            for r in event_range:
                if r.end - r.begin >= output_duration:
                    assert r.begin <= flare_peak < r.end
                    assert region_start <= flare_peak < region_end
                    assert r.begin < r.end
                    assert r.begin - input_duration >= region_start
                    '''larger_peak_during_range = [f for f in flares if
                                                f["fl_peakflux"] > _class_to_flux(r.data[0]) * 1.1 and
                                                f["peaktime"] >= r.begin and
                                                f["peaktime"] <= r.end and
                                                len(goes_global[(goes_global.index >= f["starttime"]) &
                                                            (goes_global.index <= f["endtime"])]) > 0
                                                ]
                    assert len(larger_peak_during_range) == 0'''

                    flare_ranges.addi(r.begin, r.end, r.data)

        # Remove range around flare from free areas
        free_ranges.chop(flare_peak - output_duration + dt.timedelta(seconds=1), flare_peak + output_duration)

    # Remove free ranges where GOES > 8e-9 but no flare happens on the entire sun
    free_ranges -= goes_noflare

    # Merge free and flare ranges
    region_ranges = free_ranges | flare_ranges

    # Remove unmapped ranges from result
    for current_chop_interval in unmapped_ranges:
        region_ranges.chop(current_chop_interval.begin, current_chop_interval.end)

        if len(region_ranges[current_chop_interval]) > 0:
            # This seems to be a bug in the library where the chop operation does nothing
            # Creating a new tree seems to fix it
            region_ranges = intervaltree.IntervalTree(set(region_ranges))
            region_ranges.chop(current_chop_interval.begin, current_chop_interval.end)

        assert len(region_ranges[current_chop_interval]) == 0

    # Remove ranges which are too small
    intervals_to_remove = [
        interval for interval in region_ranges
        if interval.end - interval.begin < output_duration
    ]
    for current_interval in intervals_to_remove:
        assert current_interval in region_ranges
        region_ranges.discardi(*current_interval)
        assert current_interval not in region_ranges

    # Add peak flux to each range
    updated_ranges = intervaltree.IntervalTree()
    for current_interval in region_ranges:
        current_class, current_peak_time = current_interval.data

        # We're specifically using the manually labeled flux here, not the GOES flux.
        if current_class == "free":
            current_peak_flux = np.float128("1e-9")
        else:
            current_peak_flux = class_to_flux(current_class) / 0.85

        # if current_peak_flux is not None:
        updated_ranges.addi(
            current_interval.begin,
            current_interval.end,
            (current_class, current_peak_time, current_peak_flux)
        )

    if len(region_ranges) != len(updated_ranges):
        logger.debug("Removed %d free intervals which had bad GOES data", len(region_ranges) - len(updated_ranges))

    logger.info("Computed ranges for AR %s", noaa_id)
    return (noaa_id, updated_ranges)

goes_classes = ['quiet','A','B','C','M','X']


def flux_to_class(f: float, only_main=False):
    'maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification'
    decade = int(min(math.floor(math.log10(f)), -4))
    sub = round(10 ** -decade * f)
    if decade < -4: # avoiding class 10
        decade += sub // 10
        sub = max(sub % 10, 1)
    main_class = goes_classes[decade + 9] if decade >= -8 else 'quiet'
    sub_class = str(sub) if main_class != 'quiet' and only_main != True else ''
    return main_class + sub_class

def class_to_flux(c: str):
    'Inverse of flux_to_class \
    Maps a flare class (e.g. B6, M, X9) to a GOES flux value'
    if c == 'quiet':
        return 1e-9
    decade = goes_classes.index(c[0])-9
    sub = float(c[1:]) if len(c) > 1 else 1
    return round(10 ** decade * sub, 10)


def _assert_active_region_time_ranges(
        ranges: Dict[int, intervaltree.IntervalTree],
        unmapped_ranges: intervaltree.IntervalTree,
        output_duration: dt.timedelta
):
    for noaa_id, region_ranges in ranges.items():
        # Check length is valid
        assert not any(interval.end - interval.begin < output_duration for interval in region_ranges)

        # Check no overlaps between free regions
        sorted_intervals = list(sorted(interval for interval in region_ranges if interval.data == "free"))
        for idx, interval in enumerate(sorted_intervals):
            # Check free regions within themselfs
            assert idx == 0 or interval.begin > sorted_intervals[idx - 1].end, \
                f"Found overlap between free intervals {interval} and {sorted_intervals[idx - 1]}"

            # Check free regions and dataset
            assert len(region_ranges[interval]) == 1

        # Check distance between flare ranges
        sorted_flare_intervals = list(sorted(interval for interval in region_ranges if interval.data != "free"))
        for idx, interval in enumerate(sorted_flare_intervals):
            assert idx == 0 \
                   or interval.begin + output_duration >= sorted_flare_intervals[idx - 1].end \
                   or interval.data <= sorted_flare_intervals[idx - 1].data, \
                f"Found overlap between dataset {interval} and {sorted_flare_intervals[idx - 1]}"

        # Check no unmapped dataset are overlapped
        for interval in region_ranges:
            assert len(unmapped_ranges[interval]) == 0, \
                f"{interval} overlaps unmapped flare ranges {unmapped_ranges[interval]}"


def sample_ranges(
        all_ranges: pd.DataFrame,
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try to roughly group all active regions by their max flare class (ignoring subclasses)
    stratified_regions = collections.defaultdict(list)
    for noaa_num, region_ranges in all_ranges.groupby(["noaa_num"]):
        max_class = region_ranges[region_ranges["type"] != "free"]["type"].max()

        if pd.isna(max_class):
            stratified_regions["free"].append(noaa_num)
        else:
            stratified_regions[max_class[:2]].append(noaa_num)

    np.random.seed(seed)

    test_regions = set()
    training_regions = set()

    # Sample flare regions
    for current_max_class, current_active_regions in sorted(stratified_regions.items(), reverse=True):
        target_indices = np.array([])

        if current_max_class == "free":
            continue

        if len(current_active_regions) < 6:
            # Take one active region with 50% probability into the test set
            if np.random.rand() < 0.5:
                target_indices = np.random.choice(len(current_active_regions), (1,))
        else:
            # Take a number of active regions
            num_region_samples = 3
            target_indices = np.random.choice(len(current_active_regions), num_region_samples)

        # Special case 12673: We want the September flare to be part of the test set
        if 12673 in current_active_regions and 12673 not in target_indices:
            np.append(target_indices, 12673)

        for idx, current_region in enumerate(current_active_regions):
            if idx in target_indices:
                test_regions.add(current_region)
            else:
                training_regions.add(current_region)

    # Sample free regions
    free_active_regions = stratified_regions["free"]
    test_free_region_indices = np.random.choice(len(free_active_regions), len(test_regions) // 4)
    for idx, current_active_region in enumerate(free_active_regions):
        if idx in test_free_region_indices:
            test_regions.add(current_active_region)
        else:
            training_regions.add(current_active_region)

    logger.info("Sampled %d test and %d training active regions", len(test_regions), len(training_regions))

    # Split range frame into test and training frames
    ranges_test = all_ranges[all_ranges["noaa_num"].isin(test_regions)]
    ranges_training = all_ranges[all_ranges["noaa_num"].isin(training_regions)]
    logger.info("Split into %d test and %d training ranges", len(ranges_test), len(ranges_training))

    assert len(all_ranges) == len(ranges_test) + len(ranges_training)

    samples_test = _sample_ranges(
        ranges_test,
        input_duration,
        output_duration,
        seed
    )

    samples_training = _sample_ranges(
        ranges_training,
        input_duration,
        output_duration,
        seed
    )

    logger.info("Sampled %d test and %d training samples", len(samples_test), len(samples_training))

    return samples_test, samples_training


def _sample_ranges(
        ranges: pd.DataFrame,
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        seed: int
) -> pd.DataFrame:
    np.random.seed(seed)

    samples = pd.DataFrame(columns=ranges.columns)

    for range_id, range_values in ranges.iterrows():
        # Maximum number of samples so that no input ranges overlap
        max_samples = 1 + (range_values.end - range_values.start - output_duration) // input_duration
        if range_values.type != "free" and range_values.type >= "M" and max_samples > 1:
            # Use at least two samples for M+ dataset (if possible)
            min_samples = 2
        else:
            min_samples = 1
        range_samples = np.random.randint(min_samples, max_samples + 1)

        input_start = range_values.start - input_duration
        total_offset = range_values.end - range_values.start - output_duration \
                             - (range_samples - 1) * input_duration
        # split total offset sum into range_samples pieces
        offset_splits = np.random.uniform(size=(range_samples-1))
        offset_splits = offset_splits / np.sum(offset_splits) * total_offset

        for sample_idx in range(range_samples):
            sample_id = f"{range_id}_{sample_idx}"
            s = input_start + np.sum(offset_splits[:sample_idx]) + input_duration * sample_idx
            samples.loc[sample_id] = (
                range_values["noaa_num"],
                s,
                s + input_duration,
                range_values.type,
                range_values.peak,
                range_values.peak_flux
            )

    return samples


def verify_sampling(
        test_samples: pd.DataFrame,
        training_samples: pd.DataFrame,
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        noaa_regions: Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]],
        goes: pd.DataFrame
):
    # TODO: Check if active regions are actually not just the same with different numbers

    logger.info("Verifying that active regions in the test and training set are mutually exclusive")
    test_region_numbers = {row.noaa_num for _, row in test_samples.iterrows()}
    training_region_numbers = {row.noaa_num for _, row in training_samples.iterrows()}
    region_number_intersection = test_region_numbers & training_region_numbers
    assert len(region_number_intersection) == 0, \
        f"Same active regions found in both test and training set: {region_number_intersection}"

    logger.info("Internally verifying test samples")
    _verify_sampling_internal(test_samples, input_duration, output_duration, noaa_regions, goes)

    logger.info("Internally verifying training samples")
    _verify_sampling_internal(training_samples, input_duration, output_duration, noaa_regions, goes)


def _verify_sampling_internal(
        samples: pd.DataFrame,
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        noaa_regions: Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]],
        goes: pd.DataFrame
):
    logger.info("Verifying sample input duration")
    for sample_id, sample_values in samples.iterrows():
        assert sample_values.end - sample_values.start == input_duration, \
            f"Sample {sample_id} ({sample_values.start} - {sample_values.end}) has invalid duration {input_duration}"

    logger.info("Verifying that peak fluxes happen after the input period")
    for sample_id, sample_values in samples.iterrows():
        assert sample_values.type != "free" or pd.isna(sample_values.peak)
        assert sample_values.type == "free" or sample_values.peak >= sample_values.end, \
            f"Sample {sample_id} peak flux ({sample_values.peak}) must happen after the input {sample_values.end}"

    logger.info("Verifying that peak fluxes happen in the output period")
    for sample_id, sample_values in samples.iterrows():
        assert sample_values.type == "free" or sample_values.peak < sample_values.end + output_duration, \
            f"Sample {sample_id} peak flux ({sample_values.peak}) must happen in output period {sample_values.end + output_duration}"

    logger.info("Verifying that inputs are fully contained in their active region")
    for sample_id, sample_values in samples.iterrows():
        region_start, region_end, _ = noaa_regions[sample_values.noaa_num]
        assert region_start <= sample_values.start, \
            f"Sample {sample_id} input start {sample_values.start} is before the corresponding region start {region_start}"

        # not necessarily the case, but the way we process events this holds true.
        assert sample_values.end + output_duration <= region_end + dt.timedelta(seconds=1), \
            f"Sample {sample_id} output end {sample_values.end + output_duration} ends after the corresponding region end {region_end + dt.timedelta(seconds=1)}"

    # Verify that there's no higher peak during a sample, from the same AR
    '''for sample_id, sample_values in samples.iterrows():
        overlapping_higher_peaks = samples.loc[(samples['noaa_num'] == sample_values.noaa_num) &
                    (samples.index != sample_id) &
                    (samples['peak'] >= sample_values['start'] + output_duration) &
                    (samples['peak'] <= sample_values['end'] + output_duration) &
                    (samples['peak_flux'] > sample_values['peak_flux'] * 1.1) # threshold
        ]
        assert len(overlapping_higher_peaks) == 0, f"During sample {sample_id} with peak {sample_values['peak_flux']}, \
        there's a higher peak value {overlapping_higher_peaks.iloc[0]['peak_flux']} from sample {overlapping_higher_peaks.index[0]}"'''


    # Very slow
    '''logger.info("Verifying that peak seems present in the GOES curve (GOES flux > peak_flux)")
    for sample_id, sample_values in samples.iterrows():
        if sample_values.peak_flux > 5e-9: # A is 1e-8
            print(sample_id)
            overlap_range = goes[(goes.index >= sample_values.peak - dt.timedelta(minutes=10)) & (goes.index <= sample_values.peak + dt.timedelta(minutes=10))]
            if len(overlap_range) > 0:
                maxGOES = overlap_range['A_FLUX'].max()
                #assert (maxGOES / sample_values.peak_flux) > 0.85, \
                #    f"Sample {sample_id} with output end {sample_values.end + output_duration} doesn't have its peak_flux represented in the GOES curve ({maxGOES:.4} vs {sample_values.peak_flux:.4})"
                if (maxGOES / sample_values.peak_flux) > 0.85:
                    logger.info(f"Sample {sample_id} with output end {sample_values.end + output_duration} doesn't have its peak_flux represented in the GOES curve ({maxGOES:.4} vs {sample_values.peak_flux:.4})")'''
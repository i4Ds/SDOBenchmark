import collections
import datetime as dt
import logging
from typing import List, Tuple, Dict

import intervaltree
import numpy as np
import pandas as pd

import dataset.util as util

logger = logging.getLogger(__name__)


def extract_events(raw_events: List[dict]) -> Tuple[List[dict], Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]]]:

    noaa_regions = dict()
    for event in raw_events:
        event['starttime'] = util.hek_date(event["event_starttime"])
        event['endtime'] = util.hek_date(event["event_endtime"])
        if event["event_type"] == "FL":
            event['peaktime'] = util.hek_date(event["event_peaktime"])
            event['fl_peakflux'] = _class_to_flux(event["fl_goescls"])

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
        goes: pd.DataFrame
) -> Dict[int, intervaltree.IntervalTree]:

    # Create interval tree of unmapped ranges for fast lookup later on
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
    goes_interval = intervaltree.IntervalTree()
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
                    goes_interval.addi(interval_start_date, prev_date)
                interval_start_date = None
        prev_flux = flux
        prev_date = date

    # tree where GOES > 8e-9 but no (mapped) dataset present
    goes_noflare = goes_interval - mapped_flares_entire_sun

    total_free_count = 0
    total_free_sum = dt.timedelta(seconds=0)

    result = dict()
    for noaa_id, (region_start, region_end, region_events) in noaa_regions.items():
        flares = list(sorted(
            (flare_event for flare_event, flare_region_id in flare_mapping if flare_region_id == noaa_id),
            key=lambda flare_event: flare_event["peaktime"]
        ))

        # Create initial free range (add 1 second to end because region_end is inclusive but intervaltree is exclusive)
        free_ranges = intervaltree.IntervalTree()
        free_ranges.addi(region_start + input_duration, region_end + dt.timedelta(seconds=1), ("free", None))

        # Process each flare
        flare_ranges = intervaltree.IntervalTree()
        for idx, flare_event in enumerate(flares):
            flare_peak = flare_event["peaktime"]
            flare_class = flare_event["fl_goescls"]

            # Calculate best case range
            # 1 second deltas are used to make sure values are correctly inclusive/exclusive in ranges
            range_min = max(flare_peak - output_duration + dt.timedelta(seconds=1), region_start + input_duration)
            range_max = min(flare_peak + output_duration, region_end + dt.timedelta(seconds=1))

            # iff there's a bigger flare around from the same AR (HEK),
            # check with GOES curve for correct cuts.
            for idx2, flare_event2 in enumerate(flares):
                if idx2 == idx:
                    continue

                # Does a flare of the same region overlap?
                if flare_event["starttime"] <= flare_event2["endtime"] and flare_event["endtime"] >= flare_event2["starttime"]:

                    # Cut where flare_event2's flare flux becomes higher than flare_event's peak_flux
                    if flare_event2["starttime"] < flare_event["endtime"]:
                        overlap_range = goes[(goes.index > flare_event2["starttime"]) & (goes.index <= flare_event["endtime"])]
                        # cut when two successive GOES values are higher than fl_peakflux
                        last_over_thresh = False
                        for (f_time, f_val) in overlap_range.itertuples():
                            if f_val > flare_event["fl_peakflux"]:
                                if last_over_thresh is False:
                                    last_over_thresh = True
                                else:
                                    range_max = f_time - dt.timedelta(seconds=30)
                                    break
                            else:
                                last_over_thresh = False
                    elif flare_event2["endtime"] > flare_event["starttime"]:
                        overlap_range = goes[(goes.index > flare_event["starttime"]) & (goes.index <= flare_event2["endtime"])]
                        overlap_range = overlap_range[::-1]
                        # cut when two successive GOES values are higher than peak_flux
                        last_over_thresh = False
                        for (f_time, f_val) in overlap_range.itertuples():
                            if f_val > flare_event["fl_peakflux"]:
                                if last_over_thresh is False:
                                    last_over_thresh = True
                                else:
                                    range_min = f_time + dt.timedelta(seconds=30)
                                    break
                            else:
                                last_over_thresh = False

            if range_max - range_min >= output_duration:
                assert range_min <= flare_peak < range_max
                assert region_start <= flare_peak < region_end
                assert range_min < range_max
                assert range_min - input_duration >= region_start

                flare_ranges.addi(range_min, range_max, (flare_class, flare_peak))

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

            if current_class == "free":
                current_peak_flux = np.float128("1e-9")
                total_free_count += 1
                total_free_sum += current_interval.end - current_interval.begin
            else:
                current_peak_flux = _class_to_flux(current_class)

            #if current_peak_flux is not None:
            updated_ranges.addi(
                current_interval.begin,
                current_interval.end,
                (current_class, current_peak_time, current_peak_flux)
            )

        if len(region_ranges) != len(updated_ranges):
            logger.debug("Removed %d free intervals which had bad GOES data", len(region_ranges) - len(updated_ranges))

        result[noaa_id] = updated_ranges

    logger.warning("Approximating peak flux for %d free ranges (total %s) as 1e-9", total_free_count, str(total_free_sum))

    return result


def _class_to_flux(goes_class: str) -> np.float128:
    scale = {
        "A": np.float128(1e-8),
        "B": np.float128(1e-7),
        "C": np.float128(1e-6),
        "M": np.float128(1e-5),
        "X": np.float128(1e-4)
    }[goes_class[0]]

    # https://www.ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf
    # "To get the true fluxes, divide the short band flux by 0.85 and divide the long band flux by 0.7"
    scale = scale / 0.85

    return scale * np.float128(goes_class[1:])


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

        assert sample_values.end + output_duration <= region_end + dt.timedelta(seconds=1), \
            f"Sample {sample_id} output end {sample_values.end + output_duration} ends after the corresponding region end {region_end + dt.timedelta(seconds=1)}"

    logger.info("Verifying that inputs don't overlap")
    prev_sample_end = dt.datetime(2000,1,1)
    samples = samples.sort_index()
    for sample_id, sample_values in samples.iterrows():
        assert sample_values.start > prev_sample_end, \
            f"Sample {sample_id} overlaps with the previous sample's input period"
        prev_sample_end = sample_values.start

    # Very slow. Max difference detected was 1 order of magnitude (i.e. 10x)
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
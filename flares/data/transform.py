import collections
import datetime as dt
import logging
from typing import List, Tuple, Dict

import intervaltree
import numpy as np
import pandas as pd

import flares.util as util

logger = logging.getLogger(__name__)


def extract_events(raw_events: List[dict]) -> Tuple[List[dict], Dict[int, List[dict]]]:
    # Extract NOAA active region events and group them by their number
    noaa_regions = dict()
    for event in raw_events:
        if event["event_type"] == "AR" and event["frm_name"] == "NOAA SWPC Observer":
            noaa_id = event["ar_noaanum"]

            if noaa_id not in noaa_regions:
                noaa_regions[noaa_id] = [event]
            else:
                noaa_regions[noaa_id].append(event)

    # Sort active regions by start and end times
    noaa_regions = {
        noaa_id: (
            min(map(lambda event: util.hek_date(event["event_starttime"]), events)),
            max(map(lambda event: util.hek_date(event["event_endtime"]), events)),
            events
        )
        for noaa_id, events in noaa_regions.items()
    }

    # Extract SWPC flares and sort by start, peak and end times
    swpc_flares = list(sorted(
        (event for event in raw_events if event["event_type"] == "FL" and event["frm_name"] == "SWPC"),
        key=lambda event: (
            util.hek_date(event["event_starttime"]),
            util.hek_date(event["event_peaktime"]),
            util.hek_date(event["event_endtime"])
        )
    ))

    return swpc_flares, noaa_regions


def map_flares(
        swpc_flares: List[dict], noaa_regions: Dict[int, List[dict]], raw_events: List[dict]
) -> Tuple[List[Tuple[dict, int]], List[dict]]:
    # First, map all SWPC flares which contain a NOAA number
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

    logger.debug("Mapped %d SWPC flares which contained a valid NOAA active region number", len(mapped_flares))
    logger.debug("Could not map %d SWPC flares in first step", len(unmapped_flares))

    # Try to match all unmapped flares to SSW events and use their NOAA number if present
    ssw_flares = filter(
        lambda event: event["event_type"] == "FL" and event["frm_name"] == "SSW Latest Events", raw_events
    )
    ssw_flares = [
        (
            util.hek_date(event["event_starttime"]),
            util.hek_date(event["event_peaktime"]),
            util.hek_date(event["event_endtime"]),
            event
        )
        for event in ssw_flares
    ]

    still_unmapped_flares = []
    for event in unmapped_flares:
        event_start = util.hek_date(event["event_starttime"])
        event_end = util.hek_date(event["event_endtime"])
        event_peak = util.hek_date(event["event_peaktime"])

        # Find matching SSW event
        match_start, match_peak, match_end, match_event = min(
            ssw_flares,
            key=lambda candidate_event: (
                abs(event_peak - candidate_event[1]),
                abs(event_end - candidate_event[2]),
                abs(event_start - candidate_event[0])
            )
        )

        # Check peak time delta
        if abs(event_peak - match_peak) > dt.timedelta(minutes=1):
            # Allow for 10 minute peak delta if start and end times are equal
            if event_start != match_start \
                    or event_end != match_end \
                    or abs(event_peak - match_peak) > dt.timedelta(minutes=10):
                still_unmapped_flares.append(event)
                continue

        # Check if event peak is inside match duration
        if not (match_start <= event_peak <= match_end):
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

    logger.debug("Mapped %d SWPC flares after second step", len(mapped_flares))
    logger.debug("Could not map %d SWPC flares after second step", len(still_unmapped_flares))

    return mapped_flares, still_unmapped_flares


def active_region_time_ranges(
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        noaa_regions: Dict[int, List[dict]],
        flare_mapping: List[Tuple[dict, int]],
        unmapped_flares: List[dict]
) -> Dict[int, intervaltree.IntervalTree]:

    # Create interval tree of unmapped ranges for fast lookup later on
    unmapped_ranges = intervaltree.IntervalTree()
    for event in unmapped_flares:
        unmapped_start = util.hek_date(event["event_starttime"])
        unmapped_end = util.hek_date(event["event_endtime"])
        unmapped_ranges.addi(unmapped_start, unmapped_end)
    unmapped_ranges.merge_overlaps()

    # TODO: Check edge-case handling (interval end is often inclusive but tree treats it as exclusive)

    result = dict()
    for noaa_id, (region_start, region_end, region_events) in noaa_regions.items():
        flares = list(sorted(
            (flare_event for flare_event, flare_region_id in flare_mapping if flare_region_id == noaa_id),
            key=lambda flare_event: util.hek_date(flare_event["event_peaktime"])
        ))

        # Create initial free range
        free_ranges = intervaltree.IntervalTree()
        free_ranges.addi(region_start, region_end, "free")

        # Process each flare
        flare_ranges = intervaltree.IntervalTree()
        for idx, flare_event in enumerate(flares):
            flare_peak = util.hek_date(flare_event["event_peaktime"])
            flare_class = flare_event["fl_goescls"]

            # Calculate best case range
            range_min = max(flare_peak - output_duration, region_start + input_duration)
            range_max = min(flare_peak + output_duration, region_end)

            # Check if any bigger flares happen before
            prev_idx = idx - 1
            while prev_idx >= 0 and util.hek_date(flares[prev_idx]["event_peaktime"]) > range_min:
                # Check if check flare is bigger than current, if yes, reduce start range
                if flares[prev_idx]["fl_goescls"] > flare_class:
                    range_min = util.hek_date(flares[prev_idx]["event_peaktime"])
                    break
                else:
                    prev_idx -= 1

            # Check if any bigger flares happen afterwards
            next_idx = idx + 1
            while next_idx < len(flares) and util.hek_date(flares[next_idx]["event_peaktime"]) < range_max:
                # Check if check flare is bigger than current, if yes, reduce start range
                if flares[next_idx]["fl_goescls"] > flare_class:
                    range_max = util.hek_date(flares[next_idx]["event_peaktime"])
                    break
                else:
                    next_idx += 1

            if range_max - range_min >= output_duration:
                assert range_min <= flare_peak <= range_max
                assert region_start <= flare_peak <= region_end
                assert range_min < range_max
                assert range_min - input_duration >= region_start

                flare_ranges.addi(range_min, range_max, flare_class)

            # Remove range around flare from free areas
            free_ranges.chop(flare_peak - output_duration, flare_peak + output_duration)

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

        result[noaa_id] = region_ranges

    return result


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

            # Check free regions and flares
            assert len(region_ranges[interval]) == 1

        # Check distance between flare ranges
        sorted_flare_intervals = list(sorted(interval for interval in region_ranges if interval.data != "free"))
        for idx, interval in enumerate(sorted_flare_intervals):
            assert idx == 0 \
                   or interval.begin + output_duration >= sorted_flare_intervals[idx - 1].end \
                   or interval.data <= sorted_flare_intervals[idx - 1].data, \
                f"Found overlap between flares {interval} and {sorted_flare_intervals[idx - 1]}"

        # Check no unmapped flares are overlapped
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

    # TODO: We want 12673 in the test set

    np.random.seed(seed)

    test_regions = set()
    training_regions = set()

    # Sample flare regions
    for current_max_class, current_active_regions in sorted(stratified_regions.items(), reverse=True):
        target_indices = []

        if current_max_class == "free":
            continue

        if len(current_active_regions) < 6:
            # Take one active region with 50% probability
            if np.random.rand() < 0.5:
                target_indices = np.random.choice(len(current_active_regions), (1,))
        else:
            # Take a number of active regions
            num_region_samples = 3
            target_indices = np.random.choice(len(current_active_regions), num_region_samples)

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
        max_samples = 1 + (range_values["end"] - range_values["start"] - output_duration) // input_duration
        current_samples = 1 if max_samples == 1 else np.random.randint(1, max_samples + 1)

        current_min_offset = range_values["start"] - input_duration
        for sample_idx in range(current_samples):
            # TODO: I think this is statistically not correct

            # Calculate maximum offset
            current_max_offset = range_values["end"] - output_duration - input_duration - (
                        current_samples - sample_idx - 1) * input_duration
            assert range_values["start"] - input_duration <= current_min_offset <= current_max_offset <= range_values[
                "end"] - input_duration - output_duration

            # Use random offset in range
            current_offset_range = (current_max_offset - current_min_offset) // dt.timedelta(seconds=1) + 1
            assert current_offset_range > 0
            current_offset_seconds = np.random.randint(0, current_offset_range)
            current_offset = current_min_offset + dt.timedelta(seconds=current_offset_seconds)
            assert range_values["start"] - input_duration <= current_offset <= current_max_offset

            sample_id = f"{range_id}_{sample_idx}"
            samples.loc[sample_id] = (
                range_values["noaa_num"], current_offset, current_offset + input_duration, range_values["type"]
            )

            # Update min offset
            current_min_offset = current_offset + input_duration

    return samples

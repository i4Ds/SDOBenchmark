import argparse
import csv
import datetime as dt
import logging
import os
from typing import Dict

import dateutil.parser
import intervaltree
import pandas as pd
import simplejson as json

import flares.util as util
from flares.data.extract import load_hek_data, load_goes_flux, goes_files
from flares.data.transform import extract_events, map_flares, active_region_time_ranges, sample_ranges, verify_sampling

DEFAULT_ARGS = {
    "start": dt.datetime(2012, 1, 1),
    "end": dt.datetime(2018, 1, 1),
    "input_hours": 12,
    "output_hours": 24,
    "seed": 726527
}

logger = logging.getLogger(__name__)

# TODO: Check edge-case handling in transform module (interval end is often inclusive but tree treats it as exclusive)
# TODO: Make sure no instrument issues are present (e.g. satellite maneuvers)
# TODO: SDO sensors collect less intensity over time, this has to be incorporated.

# TODO: Some active regions might still produce flares which are not detected by SWPC
# TODO: Check if SSW data is actually reliable

# TODO: Is the Mt. Wilson Class relevant for sampling?
# TODO: Make sure sampling makes actual sense

# TODO: Should JPEG2000 or FITS images be used?
# TODO: Don't subsequent input ranges in the test set skew the results due to being redundant in the prediction?
# TODO: How should data be subsampled (especially the GOES curve for non-flaring samples)
# TODO: What cadence for input should be chosen?

# TODO: Handle merging and splitting active regions for test/training split
# TODO: Make sure no active regions behind the sun are reported
# TODO: NOAA active regions might change numbers while still being the same AR, this has to be checked to avoid training/test overlaps
# TODO: Check if active regions overlap each other to avoid duplicates


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    util.configure_logging(log_level)

    path_helper = util.PathHelper(args.directory)

    load_raw(path_helper.raw_directory, args.start, args.end)

    date_suffix = f"{args.start.strftime(util.HEK_DATE_FORMAT)}_{args.end.strftime(util.HEK_DATE_FORMAT)}"
    transform_raw(
        dt.timedelta(hours=args.input_hours),
        dt.timedelta(hours=args.output_hours),
        path_helper.raw_directory,
        path_helper.intermediate_directory,
        date_suffix,
        args.seed
    )


def load_raw(output_directory: str, start: dt.datetime, end: dt.datetime):
    logger.info("Loading raw data")

    if not os.path.isdir(output_directory):
        logger.info("Creating output directory %s", output_directory)
        os.makedirs(output_directory, exist_ok=False)

    date_suffix = f"{start.strftime(util.HEK_DATE_FORMAT)}_{end.strftime(util.HEK_DATE_FORMAT)}"

    # GOES flux
    goes_raw_path = os.path.join(output_directory, f"goes")
    logger.info("Loading GOES flux to %s", goes_raw_path)

    os.makedirs(goes_raw_path, exist_ok=True)

    for target_file_name, current_date in goes_files(start, end):
        target_file_path = os.path.join(goes_raw_path, target_file_name)

        if not os.path.exists(target_file_path):
            raw_flux_data = load_goes_flux(current_date)
            if raw_flux_data is not None:
                with open(target_file_path, "w") as f:
                    f.write(raw_flux_data)

    logger.info("Loaded GOES flux")

    # HEK events
    events_raw_path = os.path.join(output_directory, f"events_{date_suffix}.json")

    if os.path.isfile(events_raw_path):
        logger.info("Using existing event list at %s", events_raw_path)
    else:
        logger.info("Event list not found, will be downloaded to %s", events_raw_path)

        with open(events_raw_path, "w") as f:
            json.dump(load_hek_data(start, end), f, iterable_as_array=True)

        logger.info("Loaded event list")


def transform_raw(
        input_duration: dt.timedelta,
        output_duration: dt.timedelta,
        input_directory: str,
        output_directory: str,
        date_suffix: str,
        seed: int
):
    logger.info("Transforming raw data")

    if not os.path.isdir(output_directory):
        logger.info("Creating output directory %s", output_directory)
        os.makedirs(output_directory, exist_ok=False)

    events_raw_path = os.path.join(input_directory, f"events_{date_suffix}.json")
    with open(events_raw_path, "r") as f:
        raw_events = json.load(f)

    swpc_flares, noaa_active_regions = extract_events(raw_events)
    logger.debug(
        "Extracted %d SWPC flares and %d (grouped) NOAA active regions", len(swpc_flares), len(noaa_active_regions)
    )

    ranges_path = os.path.join(output_directory, f"ranges_{date_suffix}.csv")

    if os.path.isfile(ranges_path):
        logger.info("Using existing ranges at %s", ranges_path)
    else:
        logger.info("Ranges not found, will be computed to %s", ranges_path)

        mapped_flares, unmapped_flares = map_flares(swpc_flares, noaa_active_regions, raw_events)
        logger.debug(
            "Created flare mapping, resulting in %d mapped and %d unmapped flares",
            len(mapped_flares), len(unmapped_flares)
        )

        ranges = active_region_time_ranges(
            input_duration, output_duration, noaa_active_regions, mapped_flares, unmapped_flares
        )
        logger.info("Computed ranges")

        _save_ranges(ranges_path, ranges)
        logger.info("Saved ranges")

    samples_test_path = os.path.join(output_directory, f"samples_test_{date_suffix}.csv")
    samples_training_path = os.path.join(output_directory, f"samples_training_{date_suffix}.csv")

    if os.path.isfile(samples_test_path) and os.path.isfile(samples_training_path):
        logger.info("Using existing test/training sets at %s and %s", samples_test_path, samples_training_path)
    else:
        logger.info(
            "Test/training sets not found, will be sampled to %s and %s", samples_test_path, samples_training_path
        )

        all_ranges = pd.read_csv(
            ranges_path,
            delimiter=";",
            index_col=0,
            parse_dates=["start", "end", "peak"]
        )

        test_samples, training_samples = sample_ranges(
            all_ranges,
            input_duration,
            output_duration,
            seed
        )
        test_samples.to_csv(samples_test_path, sep=";")
        training_samples.to_csv(samples_training_path, sep=";")
        logger.info("Sampled test/training sets")

    logger.info("Verifying sampling")
    test_samples = pd.read_csv(
        samples_test_path,
        delimiter=";",
        index_col=0,
        parse_dates=["start", "end", "peak"]
    )
    training_samples = pd.read_csv(
        samples_training_path,
        delimiter=";",
        index_col=0,
        parse_dates=["start", "end", "peak"]
    )
    verify_sampling(test_samples, training_samples, input_duration, output_duration, noaa_active_regions)
    logger.info("Sampling verified successfully")


def _save_ranges(output_path: str, ranges: Dict[int, intervaltree.IntervalTree]):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        # Header
        writer.writerow(("id", "noaa_num", "start", "end", "type", "peak"))

        for noaa_num, region_ranges in ranges.items():
            for interval in region_ranges:
                current_id = _range_id(noaa_num, interval)
                current_class, current_peak = interval.data

                writer.writerow((
                    current_id,
                    noaa_num,
                    interval.begin.strftime(util.HEK_DATE_FORMAT),
                    interval.end.strftime(util.HEK_DATE_FORMAT),
                    current_class,
                    current_peak.strftime(util.HEK_DATE_FORMAT) if current_peak is not None else None
                ))


def _range_id(noaa_num: int, interval: intervaltree.Interval):
    return f"{noaa_num}_{interval.begin:%Y_%m_%d_%H_%M_%S}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="Output directory"
    )
    parser.add_argument(
        "--start", default=DEFAULT_ARGS["start"], type=dateutil.parser.parse, help="First date and time (inclusive)"
    )
    parser.add_argument(
        "--end", default=DEFAULT_ARGS["end"], type=dateutil.parser.parse, help="Last date and time (exclusive)"
    )
    parser.add_argument(
        "--input-hours", default=DEFAULT_ARGS["input_hours"], type=int, help="Number of hours for input"
    )
    parser.add_argument(
        "--output-hours", default=DEFAULT_ARGS["output_hours"], type=int, help="Number of hours for output"
    )
    parser.add_argument(
        "--seed", default=DEFAULT_ARGS["seed"], type=int, help="Seed which is used for test/training sampling"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enabled debug logging"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

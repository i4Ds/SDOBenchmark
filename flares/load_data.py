import argparse
import csv
import datetime as dt
import logging
import os
from typing import Dict

import dateutil.parser
import intervaltree
import simplejson as json

import flares.util as util
from flares.data.extract import load_hek_data, load_goes_flux, goes_files
from flares.data.transform import extract_events, map_flares, active_region_time_ranges

DEFAULT_ARGS = {
    "start": dt.datetime(2012, 1, 1),
    "end": dt.datetime(2018, 1, 1),
    "input_hours": 12,
    "output_hours": 24
}

logger = logging.getLogger(__name__)


# TODO: Some active regions might still produce flares which are not detected by SWPC
# TODO: Make sure no instrument issues are present (e.g. satellite maneuvers)
# TODO: Could also use SPoCA ARs to get way more data
# TODO: Check if active regions overlap each other to avoid duplicates
# TODO: Check if SSW data is actually reliable
# TODO: SDO sensors collect less intensity over time, this has to be incorporated.
# TODO: What cadence for input should be chosen?
# TODO: Is it a problem that some images of a prediction period are used as inputs?
# TODO: Should SRS be used instead of SSW?
# TODO: How should data be subsampled (especially the GOES curve for non-flaring samples)


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
        date_suffix
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
        date_suffix: str
):
    logger.info("Transforming raw data")

    if not os.path.isdir(output_directory):
        logger.info("Creating output directory %s", output_directory)
        os.makedirs(output_directory, exist_ok=False)

    events_raw_path = os.path.join(input_directory, f"events_{date_suffix}.json")
    ranges_path = os.path.join(output_directory, f"ranges_{date_suffix}.csv")

    if os.path.isfile(ranges_path):
        logger.info("Using existing ranges at %s", ranges_path)
    else:
        logger.info("Ranges not found, will be computed to %s", ranges_path)

        with open(events_raw_path, "r") as f:
            raw_events = json.load(f)

        swpc_flares, noaa_active_regions = extract_events(raw_events)
        logger.debug(
            "Extracted %d SWPC flares and %d (grouped) NOAA active regions", len(swpc_flares), len(noaa_active_regions)
        )

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


def _save_ranges(output_path: str, ranges: Dict[int, intervaltree.IntervalTree]):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        # Header
        writer.writerow(("id", "noaa_num", "start", "end", "type"))

        for noaa_num, region_ranges in ranges.items():
            for interval in region_ranges:
                current_id = _range_id(noaa_num, interval)
                writer.writerow((
                    current_id,
                    noaa_num,
                    interval.begin.strftime(util.HEK_DATE_FORMAT),
                    interval.end.strftime(util.HEK_DATE_FORMAT),
                    interval.data
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
        "--debug", action="store_true", help="Enabled debug logging"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

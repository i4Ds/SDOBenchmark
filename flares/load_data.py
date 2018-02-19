import argparse
import datetime as dt
import logging
import os

import dateutil.parser
import simplejson as json

import flares.util as util
from flares.data.load import HEK_DATE_FORMAT, load_hek_data, load_goes_flux

# TODO: Think about module structure once again

DEFAULT_ARGS = {
    "start": dt.datetime(2012, 1, 1),
    "end": dt.datetime(2018, 1, 1)
}

logger = logging.getLogger(__name__)


def main(args):
    path_helper = util.PathHelper(args.directory)
    load_raw(path_helper.raw_directory, args.start, args.end)


def load_raw(output_directory: str, start: dt.datetime, end: dt.datetime):
    logger.info("Loading raw data")

    if not os.path.isdir(output_directory):
        logger.info("Creating output directory %s", output_directory)
        os.makedirs(output_directory, exist_ok=False)

    date_suffix = f"{start.strftime(HEK_DATE_FORMAT)}_{end.strftime(HEK_DATE_FORMAT)}"

    # GOES flux
    goes_raw_path = os.path.join(output_directory, f"goes")
    logger.info("Loading GOES flux to %s", goes_raw_path)

    os.makedirs(goes_raw_path, exist_ok=True)

    for current_date in util.date_range(start, end):
        date_str = current_date.strftime("%Y%m%d")
        target_file_name = f"g15_xrs_2s_{date_str}_{date_str}.csv"
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

    return parser.parse_args()


if __name__ == "__main__":
    util.configure_logging()

    main(parse_args())

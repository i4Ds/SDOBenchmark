import argparse
import datetime as dt
import simplejson as json
import os

import logging

import flares.util as util
from flares.data.load import HEK_DATE_FORMAT, load_hek_data

# TODO: Think about module structure once again

DEFAULT_ARGS = {
    "start": dt.datetime(2012, 1, 1),
    "end": dt.datetime(2018, 1, 1)
}

logger = logging.getLogger(__name__)


def main(args):
    # TODO: Use path helper or smth like that
    base_directory = os.path.abspath(os.path.expanduser(args.directory))
    raw_dir = os.path.join(base_directory, "raw")
    load_raw(raw_dir, args.start, args.end)


def load_raw(output_directory: str, start: dt.datetime, end: dt.datetime):
    logger.info("Loading raw data")

    if not os.path.isdir(output_directory):
        logger.info("Creating output directory %s", output_directory)
        os.makedirs(output_directory, exist_ok=False)

    date_suffix = f"{start.strftime(HEK_DATE_FORMAT)}_{end.strftime(HEK_DATE_FORMAT)}"
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
    # TODO: Date parsing
    # TODO: Fix arg names
    parser.add_argument("directory", help="Output directory")
    parser.add_argument("--start", default=DEFAULT_ARGS["start"], help="First date and time (inclusive)")
    parser.add_argument("--end", default=DEFAULT_ARGS["end"], help="Last date and time (exclusive)")

    return parser.parse_args()


if __name__ == "__main__":
    util.configure_logging()

    args = parse_args()

    main(args)

import datetime as dt
import logging
import os
from typing import Iterable

HEK_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


class PathHelper(object):
    def __init__(self, data_directory: str):
        """
        Create a new path helper from the given data directory.
        The root data directory will be expanded and converted into an absolute path.
        :param data_directory: Data root directory
        """
        self._data_directory = os.path.abspath(os.path.expanduser(data_directory))

    @property
    def data_directory(self):
        return self._data_directory

    @property
    def raw_directory(self):
        return os.path.join(self._data_directory, "raw")


def date_range(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[dt.date]:
    for day_offset in range((end_datetime - start_datetime).days):
        yield (start_datetime + dt.timedelta(days=day_offset)).date()


def hek_date(date_string: str) -> dt.datetime:
    return dt.datetime.strptime(date_string, HEK_DATE_FORMAT)

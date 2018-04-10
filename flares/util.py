import datetime as dt
import logging
import os
from typing import Iterable, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

HEK_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

CHANNELS = {'94': 8, '131': 9, '171': 10, '193': 11,
            '211': 12, '304': 13, '335': 14,
            '1600': 15, '1700': 16, '4500': 17,
            'continuum': 18, 'magnetogram': 19} #HMI

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


class PathHelper(object):
    def __init__(self, data_directory: str, fits_directory: Optional[str]):
        """
        Create a new path helper from the given data directory.
        The root data directory will be expanded and converted into an absolute path.
        :param data_directory: Data root directory
        """
        self._data_directory = os.path.abspath(os.path.expanduser(data_directory))
        self._fits_directory = os.path.abspath(os.path.expanduser(fits_directory)) if fits_directory else None

    @property
    def data_directory(self):
        return self._data_directory

    @property
    def raw_directory(self):
        return os.path.join(self._data_directory, "raw")

    @property
    def intermediate_directory(self):
        return os.path.join(self._data_directory, "intermediate")

    @property
    def output_directory(self):
        return os.path.join(self._data_directory, "output")

    @property
    def fits_directory(self):
        return os.path.join(self._fits_directory if self._fits_directory is not None else self._data_directory, "output")


def date_range(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[dt.date]:
    for day_offset in range((end_datetime - start_datetime).days + 1):
        yield (start_datetime + dt.timedelta(days=day_offset)).date()


def hek_date(date_string: str) -> dt.datetime:
    return dt.datetime.strptime(date_string, HEK_DATE_FORMAT)

def requests_retry_session(
    retries=10,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
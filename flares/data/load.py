import datetime as dt
import logging
import multiprocessing
import os
import re
import shutil
import urllib.request
from typing import Tuple, List

import drms
import pandas as pd

logger = logging.getLogger(__name__)


def sample_path(sample_id: str, output_directory: str) -> str:
    return os.path.join(output_directory, sample_id)


class RequestSender(object):
    SERIES_NAMES = (
        # TODO: HMI
        "aia.lev1_vis_1h",
        "aia.lev1_uv_24s",
        "aia.lev1_euv_12s"
    )

    def __init__(self, output_queue: multiprocessing.Queue, notify_email: str, cadence_hours: int):
        self._output_queue = output_queue
        self._notify_email = notify_email
        self._cadence_hours = cadence_hours

    def __call__(self, sample_input: Tuple[str, pd.Series]):
        sample_id, sample_values = sample_input
        logger.debug("Requesting data for sample %s", sample_id)

        try:
            # Perform request and provide URLs as result
            request_urls = self._perform_request(sample_id, sample_values.start, sample_values.end)
            self._output_queue.put((sample_id, request_urls))
        except Exception as e:
            logger.error("Requesting data for sample %s failed (is skipped): %s", sample_id, e)

    def _perform_request(self, sample_id: str, start: dt.datetime, end: dt.datetime) -> List[str]:
        client = drms.Client(email=self._notify_email)
        input_hours = (end - start) // dt.timedelta(hours=1)

        # Submit requests
        requests = []
        for series_name in self.SERIES_NAMES:
            query = f"{series_name}[{start:%Y.%m.%d_%H:%M:%S_TAI}/{input_hours}h@{self._cadence_hours}h]{{image}}"
            requests.append(client.export(query, method="url_quick", protocol="as-is"))

        # Wait for all requests if they have to be processed
        urls = []
        for request in requests:
            if request.id is not None:
                # Actual request had to be made, wait for result
                logger.info("As-is data not available for sample %s, created request %s", sample_id, request.id)
                request.wait()

            # If the request failed, an exception will be thrown at this point
            for _, url_row in request.urls.iterrows():
                urls.append((url_row.record, url_row.url))

            assert request.status == 0

        return urls


class ImageLoader(object):
    RECORD_PARSE_REGEX = re.compile(r"^.+\[(.+)\]\[(.+)\].+$")
    RECORD_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, output_directory: str):
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._output_directory = output_directory

    def __call__(self, *args, **kwargs):
        logging.debug("Image loader started")
        while True:
            current_input = self._input_queue.get()

            # Check if done
            if current_input is None:
                break

            sample_id, records = current_input
            logger.debug("Downloading images of sample %s", sample_id)

            sample_directory = os.path.join(self._output_directory, sample_id)
            try:
                # Download image
                self._download_images(sample_directory, records)

                # Enqueue next work item
                self._output_queue.put(sample_id)

            except Exception as e:
                logger.error("Error while downloading data for sample %s (is skipped): %s", sample_id, e)

                # Delete sample directory because it contains inconsistent data
                shutil.rmtree(sample_directory, ignore_errors=True)

    def _download_images(self, sample_directory: str, records: List[Tuple[str, str]]):
        fits_directory = os.path.join(sample_directory, "_fits_temp")
        os.makedirs(fits_directory)

        for record, url in records:
            # TODO: This does not work with HMI
            record_match = self.RECORD_PARSE_REGEX.match(record)

            if record_match is None:
                raise Exception(f"Invalid record format '{record}'")

            record_date_raw, record_wavelength = record_match.groups()
            record_date = dt.datetime.strptime(record_date_raw, self.RECORD_DATE_FORMAT)

            output_file_name = f"{record_date:%Y-%m-%dT%H%M%S}_{record_wavelength}.fits"
            urllib.request.urlretrieve(url, os.path.join(fits_directory, output_file_name))

        logger.debug("Downloaded %d files to %s", len(records), fits_directory)


class OutputProcessor(object):
    def __init__(self, input_queue: multiprocessing.Queue, output_directory: str):
        self._input_queue = input_queue
        self._output_directory = output_directory

    def __call__(self, *args, **kwargs):
        logging.debug("Output processor started")
        while True:
            sample_id = self._input_queue.get()

            # Check if done
            if sample_id is None:
                break

            logger.debug("Processing sample %s", sample_id)

            sample_directory = os.path.join(self._output_directory, sample_id)
            fits_directory = os.path.join(sample_directory, "_fits_temp")

            try:
                # Process output
                self._process_output(sample_id, fits_directory, sample_directory)
            except Exception as e:
                logger.error("Error while processing data for sample %s (is skipped): %s", sample_id, e)

                # Delete sample directory because it contains inconsistent data
                shutil.rmtree(sample_directory, ignore_errors=True)
            finally:
                # Delete fits directory in any case to avoid space issues
                shutil.rmtree(fits_directory, ignore_errors=True)

    @classmethod
    def _process_output(cls, sample_id: str, input_directory: str, output_directory: str):
        logger.debug("Created sample %s output", sample_id)

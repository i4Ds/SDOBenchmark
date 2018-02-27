import datetime as dt
import logging
import multiprocessing
import os
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
            self._output_queue.put(request_urls)
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
    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
        self._input_queue = input_queue
        self._output_queue = output_queue

    def __call__(self, *args, **kwargs):
        logging.debug("Image loader started")
        while True:
            records = self._input_queue.get()

            # Check if done
            if records is None:
                break

            # Download image
            self._download_images(records)

            # Enqueue next work item
            # TODO: Use actual data
            self._output_queue.put(1)

    @classmethod
    def _download_images(cls, records: List[Tuple[str, str]]):
        # TODO: Perform actual work
        from time import sleep
        import random
        sleep(random.uniform(8, 20))
        logger.info("Downloaded images (%d)", len(records))


class OutputProcessor(object):
    def __init__(self, input_queue: multiprocessing.Queue):
        self._input_queue = input_queue

    def __call__(self, *args, **kwargs):
        logging.debug("Output processor started")
        while True:
            current_input = self._input_queue.get()

            # Check if done
            if current_input is None:
                break

            # Process output
            self._process_output()

    @classmethod
    def _process_output(cls):
        # TODO: Perform actual work
        from time import sleep
        import random
        sleep(random.uniform(9, 18))
        logger.info("Processed output")

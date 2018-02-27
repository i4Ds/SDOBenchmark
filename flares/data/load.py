import logging
import multiprocessing
import os
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def sample_path(sample_id: str, output_directory: str) -> str:
    return os.path.join(output_directory, sample_id)


class RequestSender(object):
    def __init__(self, output_queue: multiprocessing.Queue):
        self._output_queue = output_queue

    def __call__(self, sample_input: Tuple[str, pd.Series]):
        sample_id, sample_values = sample_input
        logger.debug("Requesting data for sample %s", sample_id)

        # Perform request
        self._perform_request()

        # Enqueue next work item
        # TODO: Use actual data
        self._output_queue.put(1)

    @classmethod
    def _perform_request(cls):
        # TODO: Perform actual work
        from time import sleep
        import random
        sleep(random.uniform(5, 9))
        logger.info("Request performed")


class ImageLoader(object):
    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
        self._input_queue = input_queue
        self._output_queue = output_queue

    def __call__(self, *args, **kwargs):
        logging.debug("Image loader started")
        while True:
            current_input = self._input_queue.get()

            # Check if done
            if current_input is None:
                break

            # Download image
            self._download_images()

            # Enqueue next work item
            # TODO: Use actual data
            self._output_queue.put(1)

    @classmethod
    def _download_images(cls):
        # TODO: Perform actual work
        from time import sleep
        import random
        sleep(random.uniform(8, 20))
        logger.info("Downloaded image")


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

import datetime as dt
from typing import Iterable

import logging
import requests

HEK_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


def load_hek_data(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[dict]:
    page = 1
    while True:
        events = requests.get("http://www.lmsal.com/hek/her", {
            "cosec": "2",  # JSON format
            "cmd": "search",
            "type": "column",
            "event_type": "fl,ar",  # Flares and active regions
            "event_starttime": start_datetime.strftime(HEK_DATE_FORMAT),
            "event_endtime": end_datetime.strftime(HEK_DATE_FORMAT),
            "event_coordsys": "helioprojective",
            "x1": "-1200",
            "x2": "1200",
            "y1": "-1200",
            "y2": "1200",
            "result_limit": "500",
            "page": page
        }).json()["result"]

        if len(events) == 0:
            break

        end_date = None
        for event in events:
            end_date = dt.datetime.strptime(event["event_endtime"], HEK_DATE_FORMAT)

            yield event

        logger.info("Loaded page %d, last date was %s", page, end_date)
        page += 1

import datetime as dt
import logging
import os
from typing import Iterable, Optional

import requests

HEK_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
GOES_BASE_URL = "https://satdat.ngdc.noaa.gov/sem/goes/data/full"
GOES_START_MARKER = os.linesep + "data:" + os.linesep

logger = logging.getLogger(__name__)


def load_hek_data(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[dict]:
    page = 1
    while True:
        response = requests.get("http://www.lmsal.com/hek/her", {
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
        })

        events = response.json()["result"]

        if len(events) == 0:
            break

        end_date = None
        for event in events:
            end_date = dt.datetime.strptime(event["event_endtime"], HEK_DATE_FORMAT)

            yield event

        logger.info("Loaded page %d, last date was %s", page, end_date)
        page += 1


def load_goes_flux(date: dt.date) -> Optional[str]:
    date_str = date.strftime("%Y%m%d")
    target_file_name = f"g15_xrs_2s_{date_str}_{date_str}.csv"
    target_url = GOES_BASE_URL + f"/{date.year}/{date.month:02}/goes15/csv/" + target_file_name

    try:
        response = requests.get(target_url)
        response.raise_for_status()

        return response.text
    except requests.HTTPError as e:
        logger.warning("HTTP error while loading %s, will be skipped: %s", target_url, e)
        return None
    except Exception as e:
        logger.warning("Unknown error while loading %s, will be skipped: %s", target_url, e)
        return None

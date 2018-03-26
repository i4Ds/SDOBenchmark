import datetime as dt
import logging
import os
from typing import Iterable, Optional, Tuple
import pandas as pd

import requests

import flares.util as util

GOES_BASE_URL = "https://satdat.ngdc.noaa.gov/sem/goes/data/full"
#GOES_START_MARKER = os.linesep + "data:" + os.linesep

logger = logging.getLogger(__name__)


def load_hek_data(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[dict]:
    page = 1
    sess = util.requests_retry_session()
    while True:
        r = sess.get("http://www.lmsal.com/hek/her", params={
            "cosec": "2",  # JSON format
            "cmd": "search",
            "type": "column",
            "event_type": "fl,ar",  # Flares and active regions
            "event_starttime": start_datetime.strftime(util.HEK_DATE_FORMAT),
            "event_endtime": end_datetime.strftime(util.HEK_DATE_FORMAT),
            "event_coordsys": "helioprojective",
            "x1": "-1200",
            "x2": "1200",
            "y1": "-1200",
            "y2": "1200",
            "result_limit": "500",
            "page": page,
            "return": "hpc_bbox,hpc_coord,event_type,intensmin,obs_meanwavel,intensmax,intensmedian,obs_channelid,ar_noaaclass,frm_name,obs_observatory,hpc_x,hpc_y,kb_archivdate,ar_noaanum,frm_specificid,hpc_radius,event_starttime,event_endtime,event_peaktime,fl_goescls,frm_daterun,fl_peakflux,fl_goescls",
            "param0": "FRM_NAME",
            "op0": "=",
            "value0": "NOAA SWPC Observer,SWPC,SSW Latest Events"
        })

        events = r.json()["result"]

        if len(events) == 0:
            break

        end_date = None
        for event in events:
            end_date = util.hek_date(event["event_endtime"])

            yield event

        logger.info("Loaded page %d, last date was %s", page, end_date)
        page += 1


def goes_files(start_datetime: dt.datetime, end_datetime: dt.datetime) -> Iterable[Tuple[str, dt.date]]:
    for current_date in util.date_range(start_datetime, end_datetime):
        date_str = current_date.strftime("%Y%m%d")
        target_file_name = f"g15_xrs_2s_{date_str}_{date_str}.csv"

        yield target_file_name, current_date


def load_goes_flux(date: dt.date) -> Optional[str]:
    date_str = date.strftime("%Y%m%d")
    target_file_name = f"g15_xrs_2s_{date_str}_{date_str}.csv"
    target_url = GOES_BASE_URL + f"/{date.year}/{date.month:02}/goes15/csv/" + target_file_name

    sess = util.requests_retry_session()
    try:
        response = sess.get(target_url)
        response.raise_for_status()

        return response.text
    except requests.HTTPError as e:
        logger.warning("HTTP error while loading %s, will be skipped: %s", target_url, e)
        return None
    except Exception as e:
        logger.warning("Unknown error while loading %s, will be skipped: %s", target_url, e)
        return None


def load_all_goes_profiles(goes_directory: str) -> pd.DataFrame:
    return pd.concat([
        _parse_goes_flux(os.path.join(goes_directory, current_file))
        for current_file in os.listdir(goes_directory)
        if os.path.exists(os.path.join(goes_directory, current_file)) and current_file.startswith("g15")
    ])

'''def goes_profile(start_datetime: dt.datetime, end_datetime: dt.datetime, goes_directory: str) -> Optional[pd.DataFrame]:
    flist = [
        _parse_goes_flux(os.path.join(goes_directory, current_file))
        for (current_file, current_date) in goes_files(start_datetime, end_datetime) #os.listdir(goes_directory)
        if os.path.exists(os.path.join(goes_directory, current_file))
    ]
    if len(flist) == 0:
        return None
    fluxes = pd.concat(flist)
    fluxes = fluxes[start_datetime:end_datetime]
    if len(fluxes) == 0:
        return None
    return fluxes

def goes_profile_fromfile(start_datetime: dt.datetime, end_datetime: dt.datetime, goes_directory: str) -> Optional[pd.DataFrame]:
    flist = [
        _parse_goes_flux(os.path.join(goes_directory, current_file))
        for (current_file, current_date) in goes_files(start_datetime, end_datetime) #os.listdir(goes_directory)
        if os.path.exists(os.path.join(goes_directory, current_file))
    ]
    if len(flist) == 0:
        return None
    fluxes = pd.concat(flist)
    fluxes = fluxes[start_datetime:end_datetime]
    if len(fluxes) == 0:
        return None
    return fluxes'''

def _parse_goes_flux(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as f:
        # Skip lines until data: label is read
        for line in f:
            if line.startswith("data:"):
                break

        return pd.read_csv(f, sep=",", parse_dates=["time_tag"], index_col="time_tag", usecols=["time_tag", "A_FLUX"])
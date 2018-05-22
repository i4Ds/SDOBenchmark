import datetime as dt
import logging
import multiprocessing
import os
import re
import shutil
import urllib.request
from urllib.request import URLError
from typing import Tuple, List, Dict
import time
import math
import warnings
import simplejson as json
import signal

import traceback

import astropy.coordinates
import astropy.time
import astropy.units as u
import drms
import numpy as np
import pandas as pd
import sunpy.coordinates
import sunpy.instr.aia
import sunpy.map
import sunpy.physics.differential_rotation
from astropy.io import fits

from PIL import Image

logger = logging.getLogger(__name__)


def sample_path(sample_id: str, output_directory: str) -> str:
    ar_nr, p = sample_id.split("_",1)
    return os.path.join(output_directory, ar_nr, p)

#TODO
def sample_exists(dir_path: str, expectedFiles=48) -> bool:
    if not os.path.isdir(dir_path):
        return False
    if len([name for name in os.listdir(dir_path) if name.endswith('.jpg')]) == expectedFiles:
        return True
    return False

def _sample_series_exists(dir_path: str, series_name: str, query_time: dt.datetime):
    if not os.path.isdir(dir_path):
        return False
    wavelengths = {
        "aia.lev1_euv_12s": ['94', '131', '171', '193', '211', '304', '355'],
        "aia.lev1_uv_24s": ['1600', '1700'],
        "aia.lev1_vis_1h": ['4500'],
        "hmi.Ic_45s": ['continuum'],
        "hmi.M_45s" : ['magnetogram']
    }
    wl = wavelengths[series_name]
    for img in [name for name in os.listdir(dir_path) if name.endswith('.jpg')]:
        img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
        # is the image one of our wavelengths we search?
        if img_wavelength in wl:
            # is the image close enough in time?
            img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")
            if abs(img_datetime - query_time) < dt.timedelta(minutes=15):
                # remove from the search wavelengths, check whether we've found the all
                wl.remove(img_wavelength)
                if len(wl) == 0:
                    return True
    return False

def _sample_image_exists(dir_path: str, wavelength: str, query_time: dt.datetime):
    if not os.path.isdir(dir_path):
        return False
    for img in [name for name in os.listdir(dir_path) if name.endswith(wavelength + '.jpg')]:
        img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
        # is the image close enough in time?
        img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")
        if abs(img_datetime - query_time) < dt.timedelta(minutes=15):
            return True
    return False


class RequestSender(object):
    """Downloads FITS URLs for later use in the ImageDownloader"""
    # http://drms.readthedocs.io/en/stable/tutorial.html
    SERIES_NAMES = (
        "aia.lev1_vis_1h",
        "aia.lev1_uv_24s",
        "aia.lev1_euv_12s",
        "hmi.Ic_45s",
        "hmi.M_45s"
    )
    HMI_KEYS = 'crlt_obs, crln_obs, ctype1, ctype2, cunit1, cunit2, crval1, crval2, cdelt1, cdelt2, crpix1, crpix2, crota2, date, instrume, wcsname, dsun_ref, rsun_ref, car_rot, obs_vr, obs_vw, obs_vn, rsun_obs, t_obs, t_rec, dsun_obs'.upper()

    RECORD_PARSE_REGEX = re.compile(r"^.+\[(.+)\]\[(.+)\].+$")
    RECORD_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    RECORD_DATE_FORMAT_HMI = "%Y.%m.%d_%H:%M:%S_TAI"

    CACHEPROCESS = None

    def __init__(self, output_queue: multiprocessing.Queue, cache_queue: multiprocessing.Queue, output_directory: str, notify_email: str, time_steps: List[int], cache_dir: str):
        self._output_queue = output_queue
        self._output_directory = output_directory
        self._notify_email = notify_email
        self._time_steps = time_steps
        self._cache_dir = cache_dir
        self._cachingQueue = cache_queue
        self._initCache()

    def __call__(self, sample_input: Tuple[str, pd.Series]):
        sample_id, sample_values = sample_input
        logger.debug("Requesting data for sample %s", sample_id)

        retries = 0
        while True:
            try:
                # Perform request and provide URLs as result
                request_urls = self._perform_request(sample_id, sample_values.start, sample_values.end)
            except Exception as e:
                retries += 1
                if retries % 100 == 0:
                    logger.info(f'Failed fetching URLs for sample %s after {retries} retries: %s', sample_id, e)
                    if isinstance(e, URLError) and isinstance(e.reason, ConnectionRefusedError):
                        logger.info('waiting for a while longer...')
                    else:
                        break
                time.sleep(0.5)
            else:
                logger.info(f'received URLs for {sample_id} after {retries} retries')
                if len(request_urls) > 0:
                    self._output_queue.put((sample_id, request_urls))
                break

    def terminate(self):
        self._cachingQueue.put(None)

    def _perform_request(self, sample_id: str, start: dt.datetime, end: dt.datetime) -> List[str]:
        client = drms.Client(email=self._notify_email, verbose=False)
        #input_hours = (end - start) // dt.timedelta(hours=1)

        fp = sample_path(sample_id, self._output_directory)

        # Submit requests
        requests = []
        for series_name in self.SERIES_NAMES:
            for hd in self._time_steps: # [] minutes after input start. (last = 10min before prediction period)
                qt = start + dt.timedelta(minutes=hd)
                if not _sample_series_exists(fp, series_name, qt):
                    query = f"{series_name}[{qt:%Y.%m.%d_%H:%M:%S_TAI}]"
                    seg = 'image'

                    if series_name.startswith('hmi.Ic_'):
                        seg = 'continuum'
                    elif series_name.startswith('hmi.M_'):
                        seg = 'magnetogram'

                    query = query + '{' + seg + '}'

                    if query in self._answersCache:
                        requests.append((self._answersCache[query], qt, query))
                    else:
                        requests.append((client.export(query, method="url_quick", protocol="as-is"), qt, query))

        # Wait for all requests if they have to be processed
        urls = []
        for request, requested_date, query in requests:
            if isinstance(request, drms.ExportRequest): # not cached
                is_specific_export = False # valid only for a short time?
                if request.id is not None:
                    # Actual request had to be made, wait for result
                    logger.debug("As-is data not available for sample %s, created request %s", sample_id, request.id)
                    is_specific_export = True
                    request.wait()

                if request.status == 4: # Empty set
                    self._cachingQueue.put((query, None))
                    continue

                result = []
                for _, url_row in request.urls.iterrows():
                    result.append((url_row.record, url_row.url))
                if not is_specific_export:
                    self._cachingQueue.put((query, result))
            else:
                if request is None:
                    continue
                result = request

            for record, url in result:
                record_match = self.RECORD_PARSE_REGEX.match(record)

                if record_match is None:
                    logger.info(f"Invalid record format '{record}'")
                    continue

                record_date_raw, record_wavelength = record_match.groups()

                is_HMI = len(record_wavelength) == 1
                record_date = dt.datetime.strptime(record_date_raw, self.RECORD_DATE_FORMAT_HMI if is_HMI else self.RECORD_DATE_FORMAT)

                if abs(record_date - requested_date) < dt.timedelta(minutes=15):
                    if not _sample_image_exists(fp, record_wavelength, requested_date):
                        extra_keys = None
                        if is_HMI:
                            extra_keys = client.query(query.split('{')[0],key=self.HMI_KEYS)
                            extra_keys = extra_keys.iloc[0]
                        urls.append((record, url, extra_keys))

        return urls

    def _initCache(self):
        if not os.path.isfile(self._cache_dir):
            with open(self._cache_dir, "w") as f:
                json.dump({}, f, iterable_as_array=True)
        with open(self._cache_dir, "r") as f:
            self._answersCache = json.load(f)
        cp = multiprocessing.Process(target=_requestCaching, args=(self._cachingQueue, self._cache_dir))
        cp.start()

def _requestCaching(q, cdir):
    with open(cdir, "r") as f:
        _answersCache = json.load(f)
    signal_received = False
    
    def signalHandler(sig, frame):
        nonlocal signal_received
        signal_received = (sig, frame)

    while True:
        answer = q.get()
        if answer is None:
            break
        _answersCache[answer[0]] = answer[1]
        s = signal.signal(signal.SIGINT, signalHandler)
        with open(cdir, "w") as f:
            json.dump(_answersCache, f, iterable_as_array=True)
        signal.signal(signal.SIGINT, s)
        if signal_received:
            s(*signal_received)


class ImageLoader(object):
    RECORD_PARSE_REGEX = re.compile(r"^.+\[(.+)\]\[(.+)\].+$")
    HMI_PARSE_REGEX = re.compile(r".+{(.+)}$")
    RECORD_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    RECORD_DATE_FORMAT_HMI = "%Y.%m.%d_%H:%M:%S_TAI"
    DATE_KEYS = ["DATE", "T_OBS", "T_REC"]

    def __init__(
            self,
            input_queue: multiprocessing.Queue,
            output_queue: multiprocessing.Queue,
            output_directory: str,
            fits_directory: str
    ):
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._output_directory = output_directory
        self._fits_directory = fits_directory

    def __call__(self, *args, **kwargs):
        logger.debug("Image loader started")
        while True:
            current_input = self._input_queue.get()
            logger.info(f'Remaining URL sets in queue: {self._input_queue.qsize()}')

            # Check if done
            if current_input is None:
                break

            sample_id, records = current_input
            logger.debug("Downloading images of sample %s", sample_id)

            # Check whether the images already exist
            #if not sample_exists(sample_path(sample_id, self._output_directory), expectedFiles=len(records)):
            fits_directory = sample_path(sample_id, self._fits_directory)
            try:
                # Download image
                self._download_images(fits_directory, records)

                # Enqueue next work item
                self._output_queue.put(sample_id)

            except Exception as e:
                logger.error("Error while downloading data for sample %s (is skipped): %s", sample_id, e)
                traceback.print_exc()
                # Delete sample directory because it contains inconsistent data
                #shutil.rmtree(sample_directory, ignore_errors=True)
            #else:
                #logger.info(f'Sample {sample_id} already exists')

    def _download_images(self, fits_directory: str, records: List[Tuple[str, str]]):
        fits_directory = os.path.join(fits_directory, "_fits_temp")
        os.makedirs(fits_directory, exist_ok=True)

        logger.debug(f'Downloading {len(records)} FITS files into {fits_directory}...')

        for record, url, extra_keys in records:
            record_match = self.RECORD_PARSE_REGEX.match(record)

            if record_match is None:
                raise Exception(f"Invalid record format '{record}'")

            record_date_raw, record_wavelength = record_match.groups()

            record_date = dt.datetime.strptime(record_date_raw, self.RECORD_DATE_FORMAT_HMI if len(
                record_wavelength) == 1 else self.RECORD_DATE_FORMAT)

            if len(record_wavelength) == 1:
                record_wavelength = self.HMI_PARSE_REGEX.match(record).groups()[0]

            output_file_name = f"{record_date:%Y-%m-%dT%H%M%S}_{record_wavelength}.fits"
            fp = os.path.join(fits_directory, output_file_name)
            if not os.path.isfile(fp): #TODO: Check for corruption, incomplete files
                retries = 0
                while True:
                    try:
                        urllib.request.urlretrieve(url, fp)
                    except Exception as e:
                        retries += 1
                        if retries % 100 == 0:
                            logger.info(f'Failed fetching FITS %s after {retries} retries: %s', url, e)
                            if isinstance(e, URLError) and isinstance(e.reason, ConnectionRefusedError):
                                logger.info('waiting for a while longer...')
                            else:
                                break
                        time.sleep(0.5)
                    else:
                        logger.info(f'{retries} retries')
                        # extend HMI Fits with extra keys
                        if extra_keys is not None:
                            try:
                                data, header = fits.getdata(fp, header=True)
                                if header['BITPIX'] == -32 or header['BITPIX'] == -64:
                                    del header['BLANK'] # https://github.com/astropy/astropy/issues/7253
                                for k in extra_keys.iteritems():
                                    if k[1] == 'Invalid KeyLink':
                                        logger.warning(f'Invalid KeyLink for {k[0]}, {fp}')
                                        continue
                                    if k[0].upper() not in self.DATE_KEYS:
                                        header[k[0]] = k[1]
                                    else:
                                        pdt = drms.to_datetime(k[1]).to_pydatetime()
                                        if pdt is not pd.NaT:
                                            header[k[0]] = pdt.strftime("%Y-%m-%dT%H%M%S")
                                fits.writeto(fp, data, header, overwrite=True)
                            except Exception as e:
                                logger.error(f"Unable to extend HMI file {fp}, removing & skipping... {e}")
                                try:
                                    os.remove(fp)
                                except Exception as e2:
                                    logger.error(f'Was unable to delete file {fp}, {e2}')
                                continue

                        break
            else:
                logger.debug(f'Already found {fp}')

        logger.debug("Downloaded %d files to %s", len(records), fits_directory)


class OutputProcessor(object):
    # see _FITS_to_image
    IMAGE_PARAMS = {
        "94": {
            'dataMin': 0.1,
            'dataMax': 800,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "131": {
            'dataMin': 0.7,
            'dataMax': 1900,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "171": {
            'dataMin': 5,
            'dataMax': 3500,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "193": {
            'dataMin': 20,
            'dataMax': 5500,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "211": {
            'dataMin': 7,
            'dataMax': 3500,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "304": {
            'dataMin': 0.1,
            'dataMax': 3500,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "335": {
            'dataMin': 0.4,
            'dataMax': 1000,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "1600": {
            'dataMin': 10,
            'dataMax': 800,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "1700": {
            'dataMin': 220,
            'dataMax': 5000,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "4500": {
            'dataMin': 4000,
            'dataMax': 20000,
            'dataScalingType': 3 # 0 - linear, 1 - sqrt, 3 - log10
        },
        "continuum": {
            'dataMin': 0,
            'dataMax': 65535,
            'dataScalingType': 0
        },
        "magnetogram": {
            'dataMin': -250,
            'dataMax': 250,
            'dataScalingType': 0
        }
    }

    OUTPUT_SHAPE = (512, 512)

    def __init__(
            self,
            input_queue: multiprocessing.Queue,
            output_directory: str,
            fits_directory: str,
            meta_data: pd.DataFrame,
            noaa_regions: Dict[int, Tuple[dt.datetime, dt.datetime, List[dict]]],
            time_steps: List[int]
    ):
        self._input_queue = input_queue
        self._output_directory = output_directory
        self._fits_directory = fits_directory
        self._meta_data = meta_data
        self._noaa_regions = noaa_regions
        self._time_steps = time_steps

    def __call__(self, *args, **kwargs):
        logger.debug("Output processor started")
        while True:
            sample_id = self._input_queue.get()
            logger.info(f'Remaining FITS sets in queue: {self._input_queue.qsize()}')

            # Check if done
            if sample_id is None:
                break

            logger.debug("Processing sample %s", sample_id)

            sample_directory = sample_path(sample_id, self._output_directory)
            fits_directory = os.path.join(sample_path(sample_id, self._fits_directory), "_fits_temp")

            try:
                # Process output
                self._process_output(sample_id, fits_directory, sample_directory)

                # Delete fits directory in any case to avoid space issues
                shutil.rmtree(fits_directory, ignore_errors=True)
                print(f'Removed directory {fits_directory}')
            except Exception as e:
                logger.error(f"Error while processing data for sample {sample_id} (is skipped): {e}")
                traceback.print_exc()
                # We'll leave directories be where an error occurred, for examination purposes

    def _process_output(self, sample_id: str, input_directory: str, output_directory: str):

        # 1. Create a time line by time steps, each (available) wavelength
        # 2. For each time step
        # 3.    For each wavelength
        # 4.        Check if image is usable (in FITS header)
        # 5.        Convert to level 1.5 data (for AIA)
        # 6.        Rotate active region position to image time
        # 7.        Cut out part of image
        # 8.    Save all cuts into numpy array

        sample_meta_data = self._meta_data.loc[sample_id]
        _, _, region_events = self._noaa_regions[sample_meta_data.noaa_num]

        # Create a list of available times per wavelength
        available_times = {wavelength: [] for wavelength in self.IMAGE_PARAMS.keys()}
        for current_file in os.listdir(input_directory):
            current_datetime_raw, current_wavelength = os.path.splitext(current_file)[0].split("_")
            current_datetime = dt.datetime.strptime(current_datetime_raw, "%Y-%m-%dT%H%M%S")
            available_times[current_wavelength].append((current_datetime, current_file))

        # Assign images to actual time steps
        time_steps = [(sample_meta_data.start + dt.timedelta(minutes=offset), dict()) for offset in self._time_steps]
        for current_wavelength, current_available_times in available_times.items():
            for current_datetime, current_file in current_available_times:
                current_step_images = min(time_steps, key=lambda step: abs(step[0] - current_datetime))[1]
                assert current_wavelength not in current_step_images, f"Are there too many FITS files in the folder..? {sample_id}, {current_wavelength}"
                current_step_images[current_wavelength] = current_file

        os.makedirs(output_directory, exist_ok=True)

        # Process each time step
        for current_datetime, current_images in time_steps:

            # Process each wavelength
            for current_wavelength, current_file in current_images.items():
                fits_file = os.path.join(input_directory, current_file)
                try:
                    # recognizes a AIAMap, but doesn't automatically result in a HMIMap for HMI Fits
                    current_map = sunpy.map.Map(fits_file)
                except Exception as e:
                    logger.error(f"Unable to load file {fits_file}, removing & skipping... {e}")
                    try:
                        os.remove(fits_file)
                    except Exception as e2:
                        logger.error(f'Was unable to delete file {fits_file}, {e2}')
                    continue

                if 'date-obs' not in current_map.meta:
                    # parse the date from the file name
                    # (current_map.date == date-obs, yet date-obs is missing for HMI)
                    current_datetime_from_file_raw, _ = os.path.splitext(current_file)[0].split("_")
                    current_map.meta['date-obs'] = dt.datetime.strptime(current_datetime_from_file_raw, "%Y-%m-%dT%H%M%S")

                if isinstance(current_map, sunpy.map.sources.AIAMap):
                    # Check if map is usable
                    if not self._is_usable(current_map):
                        logger.warning("Discarding wavelength %s for sample %s", current_wavelength, sample_id)
                        continue

                    # Convert to level 1.5
                    if current_map.processing_level != 1.5:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                current_map = sunpy.instr.aia.aiaprep(current_map)
                        except Exception as e:
                            logger.error(f'aia_prep failed: {fits_file}, {e}')
                            continue
                else:
                    # custom written hmiprep, see:
                    # https://github.com/sunpy/sunpy/issues/1697, https://github.com/sunpy/sunpy/issues/2331
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            hmi_scale_factor = current_map.scale.axis1 / (0.6 * u.arcsec)
                            current_map = current_map.rotate(recenter=True, scale=hmi_scale_factor.value, missing=0.0)
                    except Exception as e:
                        logger.error(f'hmi_prep failed: {fits_file}, {e}')
                        continue

                observation_date = current_map.date
                # formerly current_map.date, which wasn't always present. Also, this is only used for image cropping, therefore some pixels of shift won't matter.
                center_x, center_y = self._find_image_center(current_map, observation_date, region_events)

                # Cut patch out of image
                # Sunpy maps assume a cartesian coordinate system which is already incorporated in the pixel conversion
                patch_start_y = center_y - self.OUTPUT_SHAPE[0] // 2
                patch_start_x = center_x - self.OUTPUT_SHAPE[1] // 2
                img = current_map.data[
                    patch_start_y:patch_start_y + self.OUTPUT_SHAPE[0],
                    patch_start_x:patch_start_x + self.OUTPUT_SHAPE[1],
                ]
                assert img.shape == self.OUTPUT_SHAPE, f'image shape is {img.shape} instead of {self.OUTPUT_SHAPE}'

                # Image processing steps
                img = self._FITS_to_image(img, current_map, current_wavelength)
                img_uint8 = (np.round(img * 255)).astype(np.uint8)

                # Save as image
                output_file_path = os.path.join(output_directory, current_datetime.strftime("%Y-%m-%dT%H%M%S") + "__" + str(current_wavelength) + ".jpg")
                im = Image.fromarray(img_uint8)
                im = im.resize((256,256), Image.BICUBIC)
                im.save(output_file_path, "jpeg")

        logger.info("Created sample %s output", sample_id)

    @classmethod
    def _is_usable(cls, target: sunpy.map.sources.AIAMap) -> bool:
        # Check header values and quality flags to be mostly sure the image is usable
        # TODO: Are those checks enough? Are there better methods to check for faulty images?
        return \
            target.meta["ACS_MODE"] == "SCIENCE" \
            and target.meta["ACS_ECLP"] != "YES" \
            and target.meta["ACS_SUNP"] == "YES" \
            and target.meta["QUALITY"] & (1 << 18) == 0  # Calibration flag

    @classmethod
    def _find_image_center(cls, current_map: sunpy.map.sources.Map, observation_date: dt.datetime, region_events: List[dict] ):
        # Find coordinates of closest active region event which started before the image
        tolerance = dt.timedelta(minutes=15)
        closest_region_event = max(
            ([event for event in region_events if event["starttime"] <= observation_date + tolerance]),
            key=lambda event: event["starttime"]
        )
        region_position = astropy.coordinates.SkyCoord(
            float(closest_region_event["hpc_x"]) * u.arcsec,
            float(closest_region_event["hpc_y"]) * u.arcsec,
            frame="helioprojective",
            obstime=closest_region_event["starttime"]
        )
        region_position_rotated = sunpy.physics.differential_rotation.solar_rotate_coordinate(
            region_position,
            observation_date
        )

        # Transform target position to pixels, in carthesian coordinates (origin bottom left)
        center_x, center_y = current_map.world_to_pixel(region_position_rotated)
        center_x, center_y = int(center_x.to_value()), int(center_y.to_value())
        assert center_x - cls.OUTPUT_SHAPE[1] / 2 >= 0, f"image out of bounds {center_x}"
        assert center_y - cls.OUTPUT_SHAPE[0] / 2 >= 0, f"image out of bounds {center_y}"
        assert center_x + cls.OUTPUT_SHAPE[1] / 2 < current_map.data.shape[1], f"image out of bounds {center_x}"
        assert center_y + cls.OUTPUT_SHAPE[0] / 2 < current_map.data.shape[0], f"image out of bounds {center_y}"

        return (center_x, center_y)

    @classmethod
    def _FITS_to_image(cls, img: np.ndarray, current_map: sunpy.map.sources.Map, wavelength: str):
        'Returns 2d array in [0,1] range'
        # Templates:
        # http://www.heliodocs.com/php/xdoc_print.php?file=$SSW/sdo/aia/idl/pubrel/aia_intscale.pro
        # https://github.com/Helioviewer-Project/jp2gen/blob/master/idl/sdo/aia/hv_aia_list2jp2_gs2.pro
        # Actually, decided to go with own clipping.
        # CRPIX recalculation from CRVAL is not necessary: http://jsoc.stanford.edu/doc/keywords/JSOC_Keywords_for_metadata.pdf
        img = np.flipud(img)
        img = img / (current_map.meta["EXPTIME"] if "EXPTIME" in current_map.meta and current_map.meta["EXPTIME"] > 0 else 1) #  normalize for exposure
        pms = cls.IMAGE_PARAMS[wavelength]
        img = np.clip(img, pms['dataMin'], pms['dataMax'])
        if pms['dataScalingType'] == 1:
            img = np.sqrt(img)
            # normalize to [0,1]
            img = (img - math.sqrt(pms['dataMin'])) / (math.sqrt(pms['dataMax']) - math.sqrt(pms['dataMin']))
        elif pms['dataScalingType'] == 3:
            img = np.log10(img)
            # normalize to [0,1]
            img = (img - math.log10(pms['dataMin'])) / (math.log10(pms['dataMax']) - math.log10(pms['dataMin']))
        else:
            img = (img - pms['dataMin']) / (pms['dataMax'] - pms['dataMin'])

        return img

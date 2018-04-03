"""Module for finding the brightness values for AIA images using the date
"""
#a change
import mov_img
from datetime import datetime, timedelta
from fetcher import fetch
from astropy.io import fits
from sunpy.map import Map
from sunpy.instr.aia import aiaprep
from sunpy.time import parse_time
import numpy as np
import multiprocessing as mp
from scipy.signal import savgol_filter
import os
import imp
import pandas as pd
from subprocess import Popen


csv_path = os.path.join(os.path.dirname(__file__), 'aia_rescaling_data.csv')
json_path = '{}json'.format(csv_path[:-3])
wavelengths = ['131','171','193','211','304','335','94']


def create_date_series(tstart):
    """
    Produce a list of dates each 1 day apart.
    
    Args:
        tstart (datetime or tuple): date to start series at
    
    Returns:
        list of dates
    
    """
    if isinstance(tstart, tuple):
        dt = datetime(*tstart)
    else:
        dt = tstart
    end = datetime.utcnow()
    step = timedelta(days=1)
    result = []
    while dt < end:
        result.append(dt)
        dt += step
    return result


# create list of datetimes and date strings
# starts at minute 1 to prevent hitting a leap second
datetime_list = create_date_series((2010,5,1, 0, 1))
date_list = [str(date.date()) for date in datetime_list]


def getFitsHdr(fle):
    """
    Get the header for a fits file
    
    Args:
        fle (str): the file name

    Returns:
        file header (dict)

    """
    f = fits.open(fle)
    hdr = f[-1].header
    f.close()
    return hdr


def get_disk_mask(data_shape, r_pix):
    """
    Returns the array mask for only the disk of the Sun

    Args:
        data_shape (tuple): shape of image data
        r_pix (float): radius of the sun in pixels

    Returns:
        numpy mask array which masks out the pixels off the disk of the sun

    """
    # x, y = np.meshgrid(*map(np.arange, data.shape), indexing='ij')
    # return (np.sqrt((x - 2047.5)**2 + (y - 2047.5)**2) > r_pix)
    nrows, ncols = data_shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = (nrows - 1) / 2, (ncols - 1) / 2
    disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > r_pix**2)
    return disk_mask


def process_med_int(fle):
    """
    Processes 1 image and extracts the median intensity on the disk
    normalized for exposure time.

    Args:
        fle (str): image file name

    Returns:
        median intensity of the solar disk normalized for exptime

    """
    amap = Map(fle)
    amap = aiaprep(amap)
    data = amap.data

    date = amap.date
    hdr = getFitsHdr(fle)
    exp_time = hdr['exptime']

    r_pix = hdr['rsun_obs'] / hdr['cdelt1'] # radius of the sun in pixels
    disk_mask = get_disk_mask(data.shape, r_pix)
    disk_data = np.ma.array(data, mask=disk_mask)
    med_int = np.ma.median(disk_data) # np.median doesn't support masking
    
    return med_int / exp_time


def no_images(fles):
    """
    Determines if no good images exist in a query

    Args:
        fles (list of strs): file names

    Returns:
        True of no images exist in a query with a quality of zero

    """
    out = len(fles) == 0
    out = out or fles[0] == ''
    out = out or getFitsHdr(fles[0])['quality'] != 0
    return out


def process_wave(wave):
    """
    Gets the median intensities for a wavelength, and the file paths

    If no *good* data is found in first 6 hours of day at 15 minutes steps,
    then the value is replaced with NaN in the series.
    Good images are those that have a "quality" rating of 0

    At the end, all NaNs are filled with the last known value until then
    Unkown values in the beginning are filled from the next known value

    Args:
        wave (str): wave to process

    Returns:
        list containing the wave str, list of filenames, and intensities

    """
    paths = pd.Series(index=date_list)
    raw = pd.Series(index=date_list)
    for date in datetime_list:
        fles = fetch(date, date + timedelta(minutes=1), wave)
        missing_data = False
        while no_images(fles):
            date += timedelta(minutes=15)
            fles = fetch(date, date + timedelta(minutes=1), wave)
            if date.hour >= 6:
                missing_data = True
                break
        # print(date)
        if not missing_data:
            index = [str(date.date())]
            fle = fles[0]
            med_int = process_med_int(fle)
            paths.loc[index] = fle
            raw.loc[index] = med_int
    paths = paths.ffill() # propagate missing values forwards
    paths = paths.bfill() # backwards. (if initial dates lack data)
    raw = raw.ffill()
    raw = raw.bfill()
    return [wave, paths, raw]


def main(compute_regression=True):
    """
    Gets all the sun intensities for all wavelengths.
    Uses multiprocessing.

    Args:
        compute_regression (bool): compute savgol regression now or later

    Returns:
        pandas dataframe containing the data

    """
    csv_dict = {}
    with mp.Pool(processes=12) as pool:
        for r in pool.imap_unordered(process_wave, wavelengths):
            wave = r[0]
            csv_dict[wave + '_paths'] = r[1]
            csv_dict[wave + '_raw'] = r[2]
            if compute_regression:
                csv_dict[wave + '_filtered'] = pd.Series(
                    savgol_filter(r[2], 301, 2), index=date_list)
    df = pd.DataFrame(csv_dict)
    return df


def main_no_mp(compute_regression=True):
    """
    Gets all the sun intensities for all wavelengths.
    Does not use multiprocessing to aid in bug fixing.
    (traceback often messed up with multiprocessing)

    Args:
        compute_regression (bool): compute savgol regression now or later

    Returns:
        Pandas dataframe containing the data
        
    """
    csv_dict = {}
    for wave in wavelengths:
        r = process_wave(wave)
        csv_dict[wave + '_paths'] = r[1]
        csv_dict[wave + '_raw'] = r[2]
        if compute_regression:
            csv_dict[wave + '_filtered'] = pd.Series(
                savgol_filter(r[2], 301, 2), index=date_list)
    df = pd.DataFrame(csv_dict)
    return df


def update_csv():
    """
    Updates the csv data.
    Looks of csv file in current directory.

    Returns:
        Pandas dataframe containing the data
    """
    if os.path.exists(csv_path):
        df = open_csv()
        # drop values that were unknown and carried forward at end
        df = df.drop(df.index[-10:])
        latest_date = parse_time(df.index[-1])
        global datetime_list
        global date_list
        datetime_list = create_date_series(latest_date
                                           + timedelta(days=1, minutes=1))
        date_list = [str(date.date()) for date in datetime_list]
        df2 = main(compute_regression=False)
        df = df.append(df2)
        for wave in wavelengths:
            df[wave + '_filtered'] = pd.Series(
                    savgol_filter(df[wave + '_raw'], 301, 2),
                    index=df.index)
    else:
        df = main()
    return df


def get_dim_factor(date, wave):
    """
    Gets the intensity scale factor for a day
    This outputs the scale factor an image's data
    should be multiplied by. Not the actual intensity.

    Args:
        date (datetime or str): datetime objects are truncated 
                                to the current day.
        wave (str): the wavelength

    Returns:
        scale factor (float)
    """
    df = open_csv()
    if not isinstance(date, str):
        date = str(date.date())
    if date > df.index[-1]:
        # Use latest known time if date ahead of csv
        date = df.index[-1]
    if date < df.index[0]:
        # Use first known time if date before start of csv data
        date = df.index[0]
    scale_factor = (df.loc['2010-05-01', wave + '_filtered']
                    / df.loc[date, wave + '_filtered'])
    return scale_factor


def get_today_factors():
    """
    Outputs a dictionary containing today's dim factors

    Returns:
        dict of of scale factors (keys are wavelengths)
    """
    factors = {}
    date = datetime.utcnow()
    for wave in wavelengths:
        factors[wave] = get_dim_factor(date, wave)
    return factors


def make_movies(out_dir='movies/'):
    """
    Uses mov_img to make a 5-second, 60-fps video of each channels's
    brightness throughout the mission with comparison between corrected
    and uncorrected images.

    Args:
        out_dir (str): ouput location

    """
    df = open_csv()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for wave in wavelengths:
        movname = out_dir + wave + '.mov'
        fits_list = df[wave + '_paths']
        decimate_factor = int(np.round(len(fits_list)/150))
        fits_list = fits_list[::decimate_factor]
        mov_img.make_movie(fits_list, movname=movname, framerate=30,
                           downscale=(32,32), side_by_side=True)


def open_csv():
    """
    Opens the aia_rescaling_data.csv file

    Returns:
        Pandas dataframe with data

    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError('Cannot find AIA data CSV file.')
    return pd.read_csv(csv_path, index_col=0)


if __name__ == '__main__':
    """
    Update the data if this module is run directly

    """
    df = update_csv()
    df.to_csv(csv_path)
    # remove unnecessary columns for json file creation
    drop_columns = [x for x in df.columns if 'raw' in x]
    drop_columns.extend([x for x in df.columns if 'paths' in x])
    df.drop(drop_columns, inplace=True, axis=1)
    df.to_json(json_path, orient='index')
    pop = Popen(['git', 'add', csv_path])
    pop.wait() # wait for previous command to execute
    pop = Popen(['git', 'add', json_path])
    pop.wait()
    pop = Popen(['git', 'commit', '-m', "'Updated regressions.'"])
    pop.wait()
    Popen(['git', 'push'])
    print('{} Regressions updated.'.format(datetime.utcnow().date()))

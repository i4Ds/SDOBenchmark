'''
Delete all the files that do not match the queries
This script was used beginning of June 2017 to purge some unwanted samples. Execute with care...
'''

import os
import pandas as pd
import datetime as dt
import shutil

base_dir = '/media/all/D4/output/2012-01-01T000000_2018-01-01T000000/'
times = [0, 7*60, 10*60+30, 11*60+50]


for phase in ['training', 'test']:
    base_path = os.path.join(base_dir, phase)

    csv_file = pd.read_csv(os.path.join(base_path, 'meta_data.csv'), sep=",", parse_dates=["start","end"], index_col="id")

    for row in csv_file.iterrows():
        ar_nr, p = row[0].split("_", 1)
        path = os.path.join(base_path, ar_nr, p)
        query_times = [row[1]['start'] + dt.timedelta(minutes=offset) for offset in times]

        # each wavelength will have a list of the 4 closest images
        closest_files = {}

        if os.path.isdir(path):
            removed = 0
            for img in [name for name in os.listdir(path) if name.endswith('.jpg')]:
                img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
                img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")

                found_matching_time = False
                for i_time, query_time in enumerate(query_times):
                    query_time_diff = abs(img_datetime - query_time)
                    if query_time_diff < dt.timedelta(minutes=15):
                        found_matching_time = True

                        if img_wavelength not in closest_files:
                            closest_files[img_wavelength] = [None, None, None, None]
                            closest_files[img_wavelength][i_time] = (query_time_diff, img)
                        else:
                            current_val = closest_files[img_wavelength][i_time]
                            if current_val is None:
                                closest_files[img_wavelength][i_time] = (query_time_diff, img)
                            else:
                                if query_time_diff >= current_val[0]:
                                    os.remove(os.path.join(path, img))
                                    removed += 1
                                else:
                                    os.remove(os.path.join(path, current_val[1]))
                                    removed += 1
                                    closest_files[img_wavelength][i_time] = (query_time_diff, img)

                if not found_matching_time:
                    os.remove(os.path.join(path, img))
                    removed += 1
            if removed > 0:
                print(f'Removed {removed} files from sample {path}')


    # but also delete all sample folders that don't exist in the csv
    for dir in os.listdir(base_path):
        sp = os.path.join(base_path, dir)
        if os.path.isdir(sp):
            for range_dir in os.listdir(sp):
                rp = os.path.join(sp, range_dir)
                if os.path.isdir(rp):
                    sample_id = dir + '_' + range_dir
                    if sample_id not in csv_file.index:
                        print(f'Sample {sample_id} does not exist!')
                        shutil.rmtree(rp)
print('Done purging')
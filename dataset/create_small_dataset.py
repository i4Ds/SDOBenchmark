'''
Create a small example dataset from the existing large dataset and meta_data files with the desired records.
This expects the large dataset to already be published.
'''

import os
import pandas as pd
from distutils.dir_util import copy_tree

publish_dir = '/media/all/D4/publish/'
publish_small_dir = '/media/all/D4/publish_small/'


for phase in ['training', 'test']:

    csv_large = pd.read_csv(os.path.join(publish_dir, phase, 'meta_data.csv'), sep=",", parse_dates=["start", "end"], index_col="id")

    # Let's check first whether you're actually ready to create a small dataset...
    missing_count = 0
    for sample_id in csv_large.index:
        sample_path = os.path.join(publish_dir, phase, *sample_id.split("_", 1))
        if not os.path.isdir(sample_path):
            print(f'{sample_path} is missing!')
            missing_count += 1

    if missing_count > 0:
        print(f'Found {missing_count} missing samples. Fix this first. Aborting...')
        continue

    # just pick every 20th line from the meta data. Yes, this is not very sophisticated...
    csv_small = csv_large.sort_values(by=['peak_flux'], ascending=False).iloc[::20].sort_values(by=['start'])
    os.makedirs(os.path.join(publish_small_dir, phase), exist_ok=True)
    csv_small.to_csv(os.path.join(publish_small_dir, phase, 'meta_data.csv'))

    for row in csv_small.iterrows():
        ar_nr, p = row[0].split("_", 1)
        publish_path = os.path.join(publish_dir, phase, ar_nr, p)

        if not os.path.isdir(publish_path):
            print('Unable to find ' + publish_path + '!')
            continue

        publish_s_path = os.path.join(publish_small_dir, phase, ar_nr, p)
        copy_tree(publish_path, publish_s_path)

print('Done copying')
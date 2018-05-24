'''
Create a small example dataset from the existing large dataset and meta_data files with the desired records.
'''

import os
import pandas as pd
from distutils.dir_util import copy_tree

publish_dir = '/media/all/D4/publish/'
publish_small_dir = '/media/all/D4/publish_small/'


for phase in ['training', 'test']:
    publish_small_path = os.path.join(publish_small_dir, phase)

    csv_file = pd.read_csv(os.path.join(publish_small_path, 'meta_data.csv'), sep=",", parse_dates=["start","end"], index_col="id")

    for row in csv_file.iterrows():
        ar_nr, p = row[0].split("_", 1)
        publish_path = os.path.join(publish_dir, phase, ar_nr, p)

        if os.path.isdir(publish_path):
            publish_s_path = os.path.join(publish_small_path, ar_nr, p)
            copy_tree(publish_path, publish_s_path)

print('Done copying')
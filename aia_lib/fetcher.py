# program for fetching list of local file names for Sun images by Matthew Pauly

# usage: 
# import fetcher
# fetcher.fetch('YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM:SS', '####,####,####,####', td=12, ds='aia.lev1_euv_12s')

# first input argument is the start time, second is the end time (string or datetime obj), and the third is a comma delimited list of channels
# there are also two named arguments td (for time delta, the minimum time between measurements)
# and ds (for data system(?), the instrument and data level we want)
# returns a list of local file names

from sunpy.time import TimeRange
import requests

def fetch(start_str, end, channel_str, td=12, ds='aia.lev1_euv_12s',
          debug=False):
        time_range = TimeRange(start_str, end)
        sec_duration = time_range.seconds
        
        if not (isinstance(start_str, str)):
                start_str = start_str.strftime('%Y-%m-%dT%H:%M:%S')
        url = 'http://jsoc.stanford.edu/cgi-bin/ajax/show_info?ds=' + ds + '[' + start_str + 'Z/' + str(sec_duration.value) + 's@' + str(td) + 's][' + channel_str + ']&q=1&P=1&seg=image'
        if debug:
                print(url)
        resp = requests.get(url)
        return resp.text.split('\n')[:-1]

if __name__ == '__main__':
        fetch('2016-02-20T13:00:00','2016-02-20T13:01:00','304,211')

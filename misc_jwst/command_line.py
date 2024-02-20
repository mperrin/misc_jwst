import astropy.time
import astropy.units as u
from . import engdb
import requests
from bs4 import BeautifulSoup



def jwstops_latest(lookback=48*u.hour):
    now = astropy.time.Time.now()
    start_time = now - lookback

    log = engdb.get_ictm_event_log(startdate=start_time)

    visit_table = engdb.visit_start_end_times(log, verbose=False, return_table=True)

    print(f"\nVisits within the previous {lookback}:\n")
    print(f"Visit ID\tStart time (UT)     \tEnd time (UT)\t   Duration [s]\tNotes")
    for row in visit_table:
        print(f"{row['visitid']}\t{row['visitstart'].iso[:-4]}\t{row['visitend'].iso[:-4]}\t{int(row['duration']):5d}\t{row['notes']}")


    latest_log_time = astropy.time.Time(log[-1]['Time'])
    print(f"\nLatest available log message ends at {latest_log_time.isot[:-4]}  ({(now-latest_log_time).to_value(u.hour):.1f} hours ago)\n")


def get_schedule_table():
    """ Get table of current observing schedules, from web-scraping the PPS postings
    """
    # Get and parse the page listing all available observing schedules. Obtain the URL of the most recent. 
    r = requests.get('https://www.stsci.edu/jwst/science-execution/observing-schedules')
    soup = BeautifulSoup(r.content, features="lxml")

    # Search on some formatting stuff that usefully tags a subset of the page markup
    divs = soup.find_all('div', attrs={'class': 'component-block container-fluid'})

    first_link = divs[2].a # First link in second division of that type
    schedule_url_1 = 'https://www.stsci.edu' + first_link.attrs['href']
    second_link = divs[2].find_all('a')[1]
    schedule_url_2 = 'https://www.stsci.edu' + second_link.attrs['href']



    # Obtain the most recent 2 schedules
    schedule1 = requests.get(schedule_url_1)
    schedule2 = requests.get(schedule_url_2)


    # Parse it into a table
    sched_table1 = astropy.io.ascii.read(schedule1.content.decode().splitlines()[2:],
                          format='fixed_width_two_line')
    sched_table2 = astropy.io.ascii.read(schedule2.content.decode().splitlines()[2:],
                          format='fixed_width_two_line')

    sched_table = astropy.table.vstack([sched_table2, sched_table1])  # last-but-one week, then current/latest week
    return sched_table

def display_schedule(sched_table, time_range=2*u.day):
    """ Print schedule table to text, with pretty formatting
    """

    short_modes = {'MIRI Medium Resolution Spectroscopy': 'MIRI MRS',
                   'MIRI Low Resolution Spectroscopy': 'MIRI LRS',
                   'MIRI Coronagraphic Imaging': 'MIRI Coron',
                   'NIRSpec MultiObject Spectroscopy': 'NRS MOS',
                   'NIRSpec IFU Spectroscopy': 'NRS IFU',
                   'NIRCam Wide Field Slitless Spectroscopy': 'NIRCam WFSS',
                   'NIRSpec Bright Object Time Series': 'NRS BOTS',
                   'NIRCam Coronagraphic Imaging': 'NIRCam Coron',
                   'NIRSpec Fixed Slit Spectroscopy': 'NRS FS',
                   'NIRCam Engineering Imaging': 'NRC Eng Img',
                   'WFSC NIRCam Fine Phasing': "WFSC",
                  }

    has_dates = [str(v).startswith("20") for v in sched_table['SCHEDULED START TIME'].value]

    sched_table[has_dates]

    marked_now = False

    now = astropy.time.Time.now()

    print(f"Showing scheduled visits within {time_range} from the current time:\n")
    for row in sched_table[has_dates]:

        long_mode = row['SCIENCE INSTRUMENT AND MODE']
        mode = short_modes.get(str(long_mode), str(long_mode))
        sched_time = astropy.time.Time(row['SCHEDULED START TIME'])

        if abs((sched_time-now)) > time_range:
            continue

        if sched_time > now and (not marked_now):
            print(f">>>>>>>>>>>   \tCurrent time       \t{now.iso[:-4]} UT \t<<<<<<<<<<<")
            marked_now = True
        print(f"{row['VISIT ID']}\t{row['VISIT TYPE']}\t{row['SCHEDULED START TIME'][:-1]}\t{mode:10s}\t{str(row['TARGET NAME'])}")
    print('\n')


def jwstops_schedule():
    sched_table = get_schedule_table()
    display_schedule(sched_table)


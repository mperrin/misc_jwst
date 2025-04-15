import astropy.time
import astropy.units as u
import numpy as np
from . import engdb
import requests
from bs4 import BeautifulSoup

import argparse

import misc_jwst.utils
from misc_jwst.visit_status import dsn_schedule

_short_modes = {'MIRI Medium Resolution Spectroscopy': 'MIRI MRS',
               'MIRI Low Resolution Spectroscopy': 'MIRI LRS',
               'MIRI Coronagraphic Imaging': 'MIRI Coron',
               'NIRCam Wide Field Slitless Spectroscopy': 'NIRCam WFSS',
               'NIRCam Engineering Imaging': 'NRC Eng Img',
               'NIRCam Coronagraphic Imaging': 'NIRCam Coron',
               'NIRISS External Calibration': 'NIRISS Ext Cal',
               'NIRISS Single-Object Slitless Spectroscopy': 'NIRISS SOSS',
               'NIRISS Wide Field Slitless Spectroscopy': 'NIRISS WFSS',
               'NIRSpec MultiObject Spectroscopy': 'NRS MOS',
               'NIRSpec IFU Spectroscopy': 'NRS IFU',
               'NIRSpec Bright Object Time Series': 'NRS BOTS',
               'NIRSpec Fixed Slit Spectroscopy': 'NRS FS',
               'WFSC NIRCam Fine Phasing': "WFSC",
               'FGS External Calibration': "FGS Ext Cal",
              }


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
    sched_tables = []
    op_package_labels = []
    for link in [first_link, second_link]:
        schedule_url = 'https://www.stsci.edu' + link.attrs['href']
        schedule = requests.get(schedule_url)
        # Parse it into a table
        sched_lines = schedule.content.decode().splitlines()
        op_package_labels.append(sched_lines[0].split()[-1])
        sched_tables.append( astropy.io.ascii.read(sched_lines[2:],
                          format='fixed_width_two_line'))

    sched_table = astropy.table.vstack(sched_tables)  # last-but-one week, then current/latest week
    sched_table.meta['op_packages'] = op_package_labels
    sched_table.meta['week2_start_time'] = sched_tables[-1][0]['SCHEDULED START TIME']  # boundary between the two weeks
    return sched_table


def display_schedule(sched_table, time_range=2*u.day):
    """ Print schedule table to text, with pretty formatting
    """

    has_dates = [str(v).startswith("20") for v in sched_table['SCHEDULED START TIME'].value]

    #sched_table[has_dates]

    marked_now = False

    now = astropy.time.Time.now()

    print(f"Showing scheduled visits within {time_range} from the current time:\n")
    print(f"VISIT ID\tVISIT TYPE      \tSCHED. FGSMAIN START\tInstr. Mode\tTarget Name")
    for row in sched_table[has_dates]:

        long_mode = row['SCIENCE INSTRUMENT AND MODE']
        mode = _short_modes.get(str(long_mode), str(long_mode))
        sched_time = astropy.time.Time(row['SCHEDULED START TIME'])

        if abs((sched_time-now)) > time_range:
            continue

        if sched_time > now and (not marked_now):
            print(f">>>>>>>>>>>   \tCurrent time       \t{now.iso[:-4]} UT \t<<<<<<<<<<<")
            marked_now = True
        print(f"{row['VISIT ID']}\t{row['VISIT TYPE']}\t{row['SCHEDULED START TIME'][:-1]}\t{mode:10s}\t{str(row['TARGET NAME'])}")
    print('\n')


def jwstops_schedule(time_range=48*u.hour):
    print(f"Time range: {time_range}")
    sched_table = get_schedule_table()
    display_schedule(sched_table, time_range=time_range)


def jwstops_visitlog(visitid, lookback=7*u.day):
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    print(f"Retrieving OSS visit log for {visitid}...")

    # First, try within the prior week
    try:
        now = astropy.time.Time.now()
        start_time = now - lookback

        log = engdb.get_ictm_event_log(startdate=start_time)
        visit_table = engdb.eventtable_extract_visit(log, visitid, verbose=False)

    except RuntimeError: # If we can't find a log in this week, look back further
        from . import mast
        start_time, end_time = mast.query_visit_time(visitid)
        log = engdb.get_ictm_event_log(startdate=start_time, enddate=end_time)
        visit_table = engdb.eventtable_extract_visit(log, visitid, verbose=False)

    for row in visit_table:
        print(row['Time'][:-4], '\t', row['Message'])


def jwstops_programstatus(program):
    """ Query the JWST Program Information page
    https://www.stsci.edu/public/jwst-program-info.html
    """
    import misc_jwst.visit_status as vs
    tab = vs.query_program_status_form(program)
    # tweak formatting a bit for display
    vis_ids = [f'{program}:{o}:{v}' for o, v in zip(tab['Observation'], tab['Visit'])]
    tab.insert(0, 'Visitid', vis_ids)
    tab.drop('Observation', axis=1, inplace=True)
    tab.drop('Visit', axis=1, inplace=True)
    print(tab)

def jwstops_guiding(visitid, lookback=7*u.day):
    import misc_jwst.guiding_analyses
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    misc_jwst.guiding_analyses.visit_guiding_sequence(visitid)


def jwstops_guiding_timeline(visitid):
    import misc_jwst.guiding_analyses
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    misc_jwst.guiding_analyses.visit_guiding_timeline(visitid)


def jwstops_guiding_performance(visitid):
    import misc_jwst.guiding_analyses
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    misc_jwst.guiding_analyses.guiding_performance_plot(visitid=visitid, save=True)
    misc_jwst.guiding_analyses.guiding_performance_jitterball(visitid=visitid, save=True)


def jwstops_durations(visitid, lookback=7*u.day):
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    print(f"Retrieving OSS visit log for {visitid}...")

    try:
        now = astropy.time.Time.now()
        start_time = now - lookback

        log = engdb.get_ictm_event_log(startdate=start_time)

        visit_table = engdb.visit_script_durations(log, visitid)
    except RuntimeError:
        # visit wasn't within the last week, so let's figure out when it was
        from . import mast
        start_time, end_time = mast.visit_start_end_times(visitid)
        log = engdb.get_ictm_event_log(startdate=start_time, enddate=end_time)
        visit_table = engdb.visit_script_durations(log, visitid)


def jwstops_deltas(lookback=48*u.hour):
    now = astropy.time.Time.now()
    start_time = now - lookback
    log = engdb.get_ictm_event_log(startdate=start_time)
    visit_table = engdb.visit_start_end_times(log, verbose=False, return_table=True)
    visit_table['VISIT ID'] = [f"{int(v[1:6])}:{int(v[6:9])}:{int(v[9:12])}" for v in list(visit_table['visitid'].value)]

    schedule = get_schedule_table()

    print(f"\nVisits within the previous {lookback}:\n")
    print(f"Visit ID\tStart time (UT)     \tSched. Start time (UT)\tTime delta [hr]\tNotes")
    for row in visit_table:
        match_index = np.where(schedule['VISIT ID'] == row['VISIT ID'])[0][0] 

        sched_start_time = astropy.time.Time(schedule[match_index]['SCHEDULED START TIME'])
        delta_time = row['visit_fgs_start'] - sched_start_time

        print(f"{row['VISIT ID']}\t{row['visit_fgs_start'].iso[:-4]}\t{sched_start_time.iso[:-4]}\t{delta_time.to_value(u.hour):+.2f}\t{row['notes']}")
    print(f"\nTimes above refer to scheduled/actual start times of FGS guide star ID+acq for each visit.\nNegative means observatory ahead of schedule, positive is behind schedule.")
    latest_log_time = astropy.time.Time(log[-1]['Time'])
    print(f"Latest available log message ends at {latest_log_time.isot[:-4]}  ({(now-latest_log_time).to_value(u.hour):.1f} hours ago)\n")


def jwstops_overview(lookback=48*u.hour):
    """ Revised top-level summary overview """
    now = astropy.time.Time.now()
    start_time = now - lookback
    log = engdb.get_ictm_event_log(startdate=start_time)
    visit_table = engdb.visit_start_end_times(log, verbose=False, return_table=True)
    visit_table['VISIT ID'] = [f"{int(v[1:6])}:{int(v[6:9])}:{int(v[9:12])}" for v in list(visit_table['visitid'].value)]

    # Retrieve schedule from the web
    schedule = get_schedule_table()
    # Drop all parallels, which don't have a scheduled time
    has_dates = [str(v).startswith("20") for v in schedule['SCHEDULED START TIME'].value]
    schedule = schedule[has_dates]

    print(f"\nVisits within the previous {lookback}:\n")
    print(f"Visit ID\tStart time (UT)     \tΔT [hr]\tInstr. Mode\tTarget\t\t\tNotes")
    for row in visit_table:
        try:
            match_index = np.where(schedule['VISIT ID'] == row['VISIT ID'])[0][0] 
            schedrow = schedule[match_index]
            sched_start_time = astropy.time.Time(schedrow['SCHEDULED START TIME'])
            long_mode = schedrow['SCIENCE INSTRUMENT AND MODE']
            mode = _short_modes.get(str(long_mode), str(long_mode))
            delta_time = row['visit_fgs_start'] - sched_start_time

            print(f"{row['VISIT ID']}\t{row['visit_fgs_start'].iso[:-4]}\t{delta_time.to_value(u.hour):+.2f}\t{mode:10s}\t{str(schedrow['TARGET NAME']):15s}\t{row['notes']}")
        except IndexError:
            print(f"{row['VISIT ID']}\t{row['visit_fgs_start'].iso[:-4]}\t??\t??\t??\t{row['notes']}")




    latest_log_time = astropy.time.Time(log[-1]['Time'])
    print(f"\t\t{latest_log_time.isot[:-4]}\t>>>>>\tLatest available log.  ({(now-latest_log_time).to_value(u.hour):.1f} hours ago)")
    marked_now = False
    for irow in range(match_index, len(schedule)):
        if irow > len(schedule):
            break
        schedrow = schedule[irow]
        sched_start_time = astropy.time.Time(schedrow['SCHEDULED START TIME'])
        long_mode = schedrow['SCIENCE INSTRUMENT AND MODE']
        mode = _short_modes.get(str(long_mode), str(long_mode))

        if sched_start_time > now and (not marked_now):
            print(f"\t\t{now.iso[:-4]}\t>>>>>\tCurrent time")
            marked_now = True
        if sched_start_time - now >  lookback:
            break

        print(f"{schedrow['VISIT ID']}\t{sched_start_time.iso[:-4]}\t    \t{mode:10s}\t{str(schedrow['TARGET NAME'])}")

    #print(f">>>>>>>>>>>   \tCurrent time       \t{now.iso[:-4]} UT \t<<<<<<<<<<<")



    print(f"\nTimes above refer to scheduled/actual start times of FGS guide star ID+acq for each visit.\nNegative ΔT means observatory ahead of schedule, positive is behind schedule.\n")


def jwstops_main():
    parser = argparse.ArgumentParser(
        description='JWST Ops tools'
    )
    #parser.add_argument('command', metavar='command', type=str, help='latest, schedule')
    parser.add_argument('-l', '--latest',  action='store_true', help='show latest (most recent) visits for which OSS logs are available')
    parser.add_argument('-s', '--schedule',  action='store_true', help='show OSS observing plan schedule')
    parser.add_argument('-t', '--time_deltas',  action='store_true', help='show timing delta between schedule and actual visit times.')
    parser.add_argument('-o', '--overview',  action='store_true', help='Ops overview; combines some of latest, schedule, and deltas')
    parser.add_argument('-P', '--schedule_plot',  action='store_true', help='Ops overview PDF plot; combines some of latest, schedule, and deltas')
    parser.add_argument('-v', '--visitlog',  help='retrieve OSS visit log for this visit (within previous week).')
    parser.add_argument('-p', '--program_status',  help='Print status of program visit execution')
    parser.add_argument('--dsn', action='store_true',  help='Print DSN communications pass schedule.')
    parser.add_argument('-g', '--guiding',  help='retrieve and plot guiding ID/ACQ/Track images for this visit (within previous week).')
    parser.add_argument('-G', '--guiding_timeline',  help='retrieve log timeline of guiding events and images for this visit.')
    parser.add_argument('--guiding_performance',  help='plot guiding performance for this visit.')
    parser.add_argument('-d', '--durations',  help='retrieve OSS visit event durations for this visit (within previous week).')
    parser.add_argument('-r', '--range',  default=48.0, help='Set time range in hours forward/back for displaying schedules. (default = 48 hours)')

    args = parser.parse_args()

    if args.latest:
        jwstops_latest(lookback=float(args.range)*u.hour)
    if args.schedule:
        jwstops_schedule(time_range=float(args.range)*u.hour)
    if args.visitlog:
        jwstops_visitlog(args.visitlog)
    if args.program_status:
        jwstops_programstatus(args.program_status)
    if args.dsn:
        dsn_schedule(lookback=float(args.range)*u.hour)

    if args.guiding:
        jwstops_guiding(args.guiding)
    if args.guiding_timeline:
        jwstops_guiding_timeline(args.guiding_timeline)
    if args.guiding_performance:
        jwstops_guiding_performance(args.guiding_performance)


    if args.durations:
        jwstops_durations(args.durations)
    if args.time_deltas:
        jwstops_deltas(lookback=float(args.range)*u.hour)
    if args.overview:
        jwstops_overview(lookback=float(args.range)*u.hour)
    if args.schedule_plot:
        # import here to avoid circular import at runtime
        from misc_jwst.schedule_plot import schedule_plot
        schedule_plot(trange=float(args.range)*u.hour)


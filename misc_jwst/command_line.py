import astropy.time
import astropy.units as u
from . import engdb


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

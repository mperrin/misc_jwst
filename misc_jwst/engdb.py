import os
import numpy as np
import argparse
from csv import reader
from datetime import datetime, timedelta, timezone
from getpass import getpass
from requests import Session
from time import sleep
import functools

import astropy.table

@functools.lru_cache
def get_ictm_event_log(startdate='2022-02-01', enddate=None, mast_api_token=None, verbose=False,
                       return_as_table=True):

    # parameters
    mnemonic = 'ICTM_EVENT_MSG'

    # constants
    base = 'https://mast.stsci.edu/jwst/api/v0.1/Download/file?uri=mast:jwstedb'
    mastfmt = '%Y%m%dT%H%M%S'
    millisec = timedelta(milliseconds=1)
    tz_utc = timezone(timedelta(hours=0))
    colhead = 'theTime'

    # establish MAST session
    session = Session()
     # set or interactively get mast token
    if not mast_api_token:
        mast_api_token = os.environ.get('MAST_API_TOKEN')
    if mast_api_token is not None:
        session.headers.update({'Authorization': f'token {mast_api_token}'})
    else:
        import warnings
        warnings.warn("Must define MAST_API_TOKEN env variable or specify mast_api_token parameter to access proprietary data")


    # fetch event messages from MAST engineering database (lags FOS EDB)
    start = datetime.fromisoformat(f'{startdate}+00:00')
    if enddate is None:
        end = datetime.now(tz=tz_utc)
    else:
        end = datetime.fromisoformat(f'{enddate}+23:59:59')

    startstr = start.strftime(mastfmt)
    endstr = end.strftime(mastfmt)
    filename = f'{mnemonic}-{startstr}-{endstr}.csv'
    url = f'{base}/{filename}'

    if verbose:
        print(f"Retrieving {url}")
    response = session.get(url)
    if response.status_code == 401:
        exit('HTTPError 401 - Check your MAST token and EDB authorization. May need to refresh your token if it expired.')
    response.raise_for_status()
    lines = response.content.decode('utf-8').splitlines()

    if return_as_table:
        return parse_eventlog_to_table(lines)
    else:
        return lines


def pretty_print_event_log(eventlog):
    # Only works on eventtable as list; unnecessary for Table format
    for value in reader(eventlog, delimiter=',', quotechar='"'):
        print(f"{value[0][0:22]:20s}\t {value[2]}")


def visit_start_end_times(eventlog, visitid=None, return_table=False, verbose=True):
    """ Find visit start and end times for all visits

    Note, 'visitstart' is the start time of the SLEW before a visit.
    'visit_fgs_start' gives the time of when the FGSMAIN ID/Acq/Track/FG began;
      this corresponds to the start time column in the public schedule on stsci.edu

    Parameters
    ----------
    return_table : bool
        Return astropy table of the outputs


    """
    # parse response (ignoring header line) and print new event messages
    vid = ''
    in_visit = False
    in_selected_visit = False

    outputs = {k: [] for k in ['visitid', 'visitstart', 'visit_fgs_start', 'visitend', 'duration', 'notes']}

    output = []
    output.append('Visit ID     | Visit Start             | Visit End'
                  '               | Duration |')
    for row in eventlog:
        msg, time = row['Message'], row['Time']
        if msg[:6] == 'VISIT ':
            if msg[-7:] == 'STARTED':
                if in_visit == True:
                    output.append(f'*** Missing end for visit {vid}')
                vstart = 'T'.join(time.split())[:-3]
                vid = msg.split()[1]
                vid_fgs_start = None

                # for debugging:
                #print("**", value)
                in_visit = True
                note = ""
            elif msg[-5:] == 'ENDED' and vid!='':
                if vid != msg.split()[1]:
                    output.append(f"Unexpected end for visit {msg} instead of {vid}")
                vend = 'T'.join(time.split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                output.append(f'{vid} | {vstart:23} | {vend:23} | '
                      f' {round(dur.total_seconds()):6}  | {note}')
                outputs['visitid'].append(vid)
                outputs['visitstart'].append(vstart)
                outputs['visit_fgs_start'].append(str(vid_fgs_start))
                outputs['visitend'].append(vend)
                outputs['duration'].append(dur.total_seconds())
                outputs['notes'].append(note)

                in_visit = False
        elif msg[:31] == f'Script terminated: {vid}':
            if msg[-5:] == 'ERROR':
                script = msg.split(':')[2]
                vend = 'T'.join(time.split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                note = f'Error in {script}'
                output.append(f'{vid} | {vstart:23} | {vend:23} | '
                      f' {round(dur.total_seconds()):6}  | {note}')
                in_visit = False
        else:
            if in_visit:
                # check for visit guide failures
                if 'FGS fixed target guide star acquisition failed on all attempts, exit FGSVERMAIN' in msg:
                    #print(f"FGS ID+Acq failed on all attempts for {vid}")
                    note = "SKIPPED. FGS ID failed all attempts"
                elif 'MIRI target locate failed' in msg:
                    note = 'MIRI target acq failed'
                elif 'NIRCam target locate failed' in msg:
                    note = 'NIRCam target acq failed'
                elif (vid_fgs_start is None) and ('Script activated' in msg) and msg.endswith('FGSMAIN'):
                    vid_fgs_start = 'T'.join(time.split())[:-3]
    else:
        if in_visit:
            output.append(f'{vid} | {vstart:23} | ongoing            | '
                  f' ongoing  | as of end of log')
            outputs['visitid'].append(vid)
            outputs['visitstart'].append(vstart)
            outputs['visit_fgs_start'].append(str(vid_fgs_start))
            outputs['visitend'].append('T'.join(time.split())[:-3])
            outputs['duration'].append(-1)
            outputs['notes'].append('Still ongoing at end of available log')





    if visitid and verbose:
        print(output[0])
        for row in output:
            if visitid in row:
                print(row)
    elif verbose:
        for row in output:
            print(row)

    if return_table:
        t = astropy.table.Table(list(outputs.values()), names = outputs.keys() )
        t['visitstart'] = astropy.time.Time(t['visitstart'])
        t['visitend'] = astropy.time.Time(t['visitend'])
        try:
            t['visit_fgs_start'] = astropy.time.Time(t['visit_fgs_start'])
        except ValueError:
            # Gracefully handle edge case, in which there may be gaps in telemetry or visits without guiding
            # This is imperfect, but in this case just swap in the visit start time for the guiding start time as a placeholder,
            # so at least the code doesn't hard stop with an exception.
            for i in range(len(t)):
                if t[i]['visit_fgs_start'] is None or t[i]['visit_fgs_start']=='None':
                    t[i]['visit_fgs_start'] = t[i]['visitstart'].isot
                    print(f'could not find FGSMAIN start time in log for visit {t[i]["visitid"]}')
            t['visit_fgs_start'] = astropy.time.Time(t['visit_fgs_start'])

        return(t)


def extract_oss_event_msgs_for_visit(eventlog, selected_visit_id, ta_only=False, verbose=False, return_text=True):
    # parse response (ignoring header line) and print new event messages
    vid = ''
    in_selected_visit = False
    in_ta = False

    messages = []

    if verbose:
        print(f"\tSearching for visit: {selected_visit_id}")
    for row in eventlog:
        msg, time = row['Message'], row['Time']

        if in_selected_visit and ((not ta_only) or in_ta) :
            if verbose:
                print(time[0:22], "\t", msg)
            if return_text:
                messages.append(time[0:22]+ "\t" + msg)

        if msg[:6] == 'VISIT ':
            if msg[-7:] == 'STARTED':
                vstart = 'T'.join(time.split())[:-3]
                vid = msg.split()[1]

                if vid==selected_visit_id:
                    if verbose:
                        print(f"VISIT {selected_visit_id} START FOUND at {vstart}")
                    in_selected_visit = True
                    if ta_only and verbose:
                        print("Only displaying TARGET ACQUISITION RESULTS:")

            elif msg[-5:] == 'ENDED' and in_selected_visit:
                assert vid == msg.split()[1]
                assert selected_visit_id  == msg.split()[1]

                vend = 'T'.join(time.split())[:-3]
                if verbose:
                    print(f"VISIT {selected_visit_id} END FOUND at {vend}")


                in_selected_visit = False
        elif msg[:31] == f'Script terminated: {vid}':
            if msg[-5:] == 'ERROR':
                script = msg.split(':')[2]
                vend = 'T'.join(time.split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                note = f'Halt in {script}'
                in_selected_visit = False
        elif in_selected_visit and msg.startswith('*'): # this string is used to mark the start and end of TA sections
            in_ta = not in_ta

    if return_text:
        return messages


def extract_oss_TA_centroids(eventlog, selected_visit_id):
    """ Return the TA centroid values from OSS
    Note, pretty sure these values from OSS are 1-based pixel coordinates - to be confirmed!

    returns (X,Y) tuple
    """

    msgs = extract_oss_event_msgs_for_visit(eventlog, selected_visit_id,
                                            ta_only=True,
                                            verbose=False, return_text=True)
    for m in msgs:
        if m.split('\t')[1].startswith("detector coord"):
            xy = ([float(p.strip('(),')) for p in m.split()[-2:]])
            return tuple(xy)
    else:
        raise RuntimeError("Could not parse TA centroid coordinates in that visit log")


def parse_eventlog_to_table(eventlog):
    """Parse an eventlog as returned from the EngDB to an astropy table, for ease of use

    """
    timestr = []
    mjd = []
    messages = []
    for value in reader(eventlog, delimiter=',', quotechar='"'):
        timestr.append(value[0])
        mjd.append(value[1])
        messages.append(value[2])

    # drop initial header row
    timestr = timestr[1:]
    mjd = np.asarray(mjd[1:], float)
    messages = messages[1:]

    # assemble into astropy table
    event_table = astropy.table.Table((timestr, mjd, messages),
                                      names=("Time", "MJD", "Message"))
    return event_table


def eventtable_extract_visit(event_table, selected_visit_id, verbose=False):
    """Find just the log message rows for a given visit"""
    vmessages = [m.startswith(f'VISIT {selected_visit_id}') for m in event_table['Message']]

    if verbose:
        print(event_table[vmessages])

    line_indices = np.where(vmessages)[0]
    if len(line_indices) == 2:
        istart, istop = line_indices
        return event_table[istart:istop+1]
    else: # visit ongoing, has not ended as of end of available log
        istart = line_indices[0]
        return event_table[istart:]


def visit_script_durations(event_table, selected_visit_id, verbose=True, return_table=False):
    visittable = eventtable_extract_visit(event_table, selected_visit_id)



    scriptevents = visittable[ [msg.startswith('Script') for msg in visittable['Message']]]


    total_visit_duration = visittable[-1]['MJD'] - visittable[0]['MJD']


    summary_durations = dict()

    def vprint(*args):
        if verbose: print(*args)

#    scriptevents = eventtable_extract_scripts(event_table, selected_visit_id)

    vprint(f"OSS Script Durations for {selected_visit_id} (total duration: {total_visit_duration*86400:.1f} s)")

    starts = dict()
    stops = dict()
    for row in scriptevents:
        if row['Message'].startswith("Script activated: "):
            key = row['Message'].split()[2]
            starts[key] = row['MJD']
        elif row['Message'].startswith("Script terminated: "):
            key = row['Message'].split()[2]

            keyparts = key.split(":")
            # Some terminate messages have extra statuses appended at the end.
            # Drop these, so the keys match up with the activate messages
            if len(keyparts) > 2:
                key = ":".join(keyparts[0:2])

            stops[key] = row['MJD']

    cumulative_time = 0

    for key in starts:
        deltatime = stops[key]-starts[key]
        vprint(f"  {key:50s}{deltatime*86400:6.1f} s")
        cumulative_time += deltatime
        activity_id, script = key.split(":")
        
        #summarize into categories: slews, FGS, TA, science obs
        if script=='SCSLEWMAIN':
            category='Slew'
        elif script in ['FGSMAIN', 'FGSMTMAIN']:
            category='FGS'
        elif 'TA' in script:
            category='TA'
        elif script[0:3] in ['NRC','NIS','NRS','MIR']:
            category = 'Parallel' if ('P' in activity_id and not activity_id.split('P')[1].startswith('00000')) else 'Science obs'
        else:
            category='Other'

        summary_durations[category] = summary_durations.get(category,0) + deltatime
         

    cumulative_time -= summary_durations.get('Parallel', 0)  # Parallels do not add to total time, by construction
    vprint(f"  Other overheads not included in the above:        {(total_visit_duration-cumulative_time)*86400:6.1f} s")
    summary_durations['Other'] += (total_visit_duration-cumulative_time)

    vprint("\nSummary by category:")
    label='Visit total'
    vprint(f"\t{label:15s}\t{total_visit_duration*86400:6.1f} s")

    categories = ['Slew', 'FGS', 'TA', 'Science obs', 'Other']
    if 'Parallel' in summary_durations:
        categories.insert(4, 'Parallel')

    parts = []
    for category in categories:
        label=category+":"
        if category not in summary_durations: # Skip if e.g. there's no TA in this visit
            parts.append(0)
            continue
        parts.append(summary_durations[category]*86400)
        vprint(f"\t{label:15s}\t{summary_durations[category]*86400:6.1f} s")
    vprint("\n")

    if return_table:
        t = astropy.table.Table([categories, parts], names = ['category', 'duration'])
        return(t)


#############################################
def main_combined():
    """ Main function for command line arguments """
    parser = argparse.ArgumentParser(
        description='Query OSS Visit Logs from JWST MAST Engineering DB'
    )
    parser.add_argument('-m', '--mast_api_token', help='MAST API token. Either set this parameter or define environment variable $MAST_API_TOKEN')
    parser.add_argument('-s', '--start_date', help='Start date for search, as YYYY-MM-DD. Defaults to 2022-07-12 if not set. ', default='2022-07-12')
    parser.add_argument('-e', '--end_date', help='End date for search, as YYYY-MM-DD. Defaults to current date if not set. ')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be more verbose for debugging')
    parser.add_argument('-f', '--full', action='store_true', help='Retrieve Full OSS log for all completed visits')
    parser.add_argument('-m', '--messages', action='store_true', help='Retrieve OSS Messages for the specified visit')
    parser.add_argument('-d', '--durations', action='store_true', help='Retrieve event durations for the specified visit')
    parser.add_argument('-t', '--ta_only', action='store_true', help='For messages, only show the Target Acquisition set of log messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be more verbose for debugging')


    args = parser.parse_args()


    eventlog = get_ictm_event_log(mast_api_token=args.mast_api_token, verbose=args.verbose,
                                  startdate=args.start_date, enddate=args.enddate)

    if args.full:
        pretty_print_event_log(eventlog)

    elif args.durations:
        event_table = eventlog_parse_to_table(eventlog)

        visit_script_durations(event_table, args.visit_id)


def main_full():
    """ Main function for command line arguments """
    parser = argparse.ArgumentParser(
        description='Get OSS ICTM Event Messages from Eng DB'
    )
    parser.add_argument('-m', '--mast_api_token', help='MAST API token. Either set this parameter or define environment variable $MAST_API_TOKEN')
    parser.add_argument('-s', '--start_date', help='Start date for search, as YYYY-MM-DD. Defaults to 2022-02-02 if not set. ', default='2022-02-02')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be more verbose for debugging')

    args = parser.parse_args()


    eventlog = get_ictm_event_log(mast_api_token=args.mast_api_token, verbose=args.verbose, startdate=args.start_date)
    pretty_print_event_log(eventlog)


def main_oss_msgs():
    """ Main function for command line arguments """
    parser = argparse.ArgumentParser(
        description='Get OSS ICTM Event Messages from Eng DB'
    )
    parser.add_argument('visit_id', type=str, help='Visit ID as a string starting with V i.e. "V01234003001"')
    parser.add_argument('-m', '--mast_api_token', help='MAST API token. Either set this parameter or define environment variable $MAST_API_TOKEN')
    parser.add_argument('-s', '--start_date', help='Start date for search, as YYYY-MM-DD. Defaults to 2022-02-02 if not set. ', default='2022-02-02')
    parser.add_argument('-t', '--ta_only', action='store_true', help='Only show the Target Acquisition set of log messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be more verbose for debugging')

    args = parser.parse_args()


    eventlog = get_ictm_event_log(mast_api_token=args.mast_api_token, verbose=args.verbose, startdate=args.start_date)

    extract_oss_event_msgs_for_visit(eventlog,  args.visit_id, verbose=args.verbose, ta_only=args.ta_only)


def main_visit_events():
    """ Main function for command line arguments """
    parser = argparse.ArgumentParser(
        description='Get OSS Event Durations from Eng DB'
    )
    parser.add_argument('visit_id', type=str, help='Visit ID as a string starting with V i.e. "V01234003001"')
    parser.add_argument('-m', '--mast_api_token', help='MAST API token. Either set this parameter or define environment variable $MAST_API_TOKEN')
    parser.add_argument('-s', '--start_date', help='Start date for search, as YYYY-MM-DD. Defaults to 2022-02-02 if not set. ', default='2022-02-02')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be more verbose for debugging')

    args = parser.parse_args()


    eventlog = get_ictm_event_log(mast_api_token=args.mast_api_token, verbose=args.verbose, startdate=args.start_date)
    event_table = eventlog_parse_to_table(eventlog)

    visit_script_durations(event_table, args.visit_id)

if __name__=="__main__":
    main()




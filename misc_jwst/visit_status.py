import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import astropy.time
import datetime
import astropy.table
import astropy.coordinates as c, astropy.units as u


def parse_plan_window(text):
    """Parse the 'Plan Windows' column information.
    Returns (start, end) if defined, else (None, None)"""

    if ((text == 'Not a candidate for the long range plan') or
        (text == 'Ready for long range planning, plan window not yet assigned')):
        return None, None
    else:
        parts = text.split(" (")[0].split(' - ')
        start_ut = parts[0]
        end_ut = parts[1]
        return start_ut, end_ut



def retrieve_status_tables(pid):
    """"""
    # Retrieve the web form, and parse the HTML
    jwst_status_form_url=f'https://www.stsci.edu/jwst-program-info/visits/?program={pid}&markupFormat=html&observatory=JWST&pi=1'
    req = requests.get(jwst_status_form_url,)
    soup = BeautifulSoup(req.content, 'html.parser')


    status_tables=[]

    # Find all tables in the HTML output
    for table in soup.findAll("table"):
        if table.findParent("table") is None:
            from io import StringIO
            status_tables.append(pd.read_html(StringIO(str(table)), header=0 )[0])

    # Iterate over the tables, and adapt them to have common format and consistent column names

    for i, table in enumerate(status_tables):
        if table['Status'][0] in ['Archived']:
            # No particular modification is needed to this table;
            # it already has Start UT and End UT columns
            table['Plan Windows'] = ''
        elif table['Status'][0] in ['Scheduling', 'Implementation']:
            # Convert 'Plan Windows' column into 'Start UT' and 'End UT'
            p = table['Plan Windows']
            table['Start UT'] = None
            table['End UT'] = None

            for i in range(len(table)):
                start, end = parse_plan_window(table['Plan Windows'][i])
                table.loc[i, 'Start UT'] = start
                table.loc[i, 'End UT'] = end

            #starts = [e.split(" (")[0].split(' - ')[0]+ " 00:00:00" for e in p]
            #ends = [e.split(" (")[0].split(' - ')[1]+" 23:59:59" for e in p]
            #
            #status_tables[1]['Start UT'] = starts
            #status_tables[1]['End UT'] = ends
            #del status_tables[1]['Plan Windows']
    return status_tables


def query_program_status_form(pid, cast_to_time=False):
    """Query the JWST Program Status Information form, and retrieve visit statuses

    That web form returns one or more tables.

    Retuns a pandas Dataframe that contains the merged, concatenated contents of all those
    tables together.
    """

    status_tables = retrieve_status_tables(pid)

    # Concatenate together the tables
    combined_table = pd.concat(status_tables, ignore_index=True)


    if cast_to_time:
        # Cast the start/end times from strings into astropy.time.Time
        start_ut = astropy.time.Time.strptime(np.array(combined_table['Start UT'], str), "%b %d, %Y %X")
        end_ut =  astropy.time.Time.strptime(np.array(combined_table['End UT'], str), "%b %d, %Y %X")
        combined_table['Start'] = start_ut
        combined_table['End'] = end_ut
        del combined_table['Start UT']
        del combined_table['End UT']

    return combined_table


def query_program_status_form_obsvis(pid, oid, vid, cast_to_time=False):
    """Query the JWST Program Status Information form, and retrieve visit status, for a fixed program,observation,visit combination 

    That web form returns one or more tables.

    Retuns a pandas Dataframe that contains the merged, concatenated contents of all those
    tables together.
    """

    status_tables = retrieve_status_tables(pid)

    # Concatenate together the tables
    combined_table = pd.concat(status_tables, ignore_index=True)


    if cast_to_time:
        # Cast the start/end times from strings into astropy.time.Time
        start_ut = astropy.time.Time.strptime(np.array(combined_table['Start UT'], str), "%b %d, %Y %X")
        end_ut =  astropy.time.Time.strptime(np.array(combined_table['End UT'], str), "%b %d, %Y %X")
        combined_table['Start'] = start_ut
        combined_table['End'] = end_ut
        del combined_table['Start UT']
        del combined_table['End UT']

    return combined_table[(combined_table['Observation'] == oid) & (combined_table['Visit'] == vid)] 


def query_program_status_form_visitid(visitid, cast_to_time=False):
    """Query the JWST Program Status Information form, and retrieve visit status, for a given input visitid

    That web form returns one or more tables.

    Display a pandas Dataframe that contains the merged, concatenated contents of all those
    tables together.
    """

    pid = int(visitid[1:6])
    oid = int(visitid[6:9])
    vid = int(visitid[9:12])
    
    status_tables = retrieve_status_tables(pid)

    # Concatenate together the tables
    combined_table = pd.concat(status_tables, ignore_index=True)


    if cast_to_time:
        # Cast the start/end times from strings into astropy.time.Time
        start_ut = astropy.time.Time.strptime(np.array(combined_table['Start UT'], str), "%b %d, %Y %X")
        end_ut =  astropy.time.Time.strptime(np.array(combined_table['End UT'], str), "%b %d, %Y %X")
        combined_table['Start'] = start_ut
        combined_table['End'] = end_ut
        del combined_table['Start UT']
        del combined_table['End UT']

    display(combined_table[(combined_table['Observation'] == oid) & (combined_table['Visit'] == vid)])

    return


def summarize_status(result_table):
    """Print out how many visits in a program are Archived, Scheduled, etc.
    """

    statcodes = list(set(result_table['Status']))
    statcodes.sort()

    for code in statcodes:
        print(f"  {code}:\t{(result_table['Status']==code).sum()} visits")

def wfsc_program_status(verbose=True):
    """Check all 4 cycle 1 WFSC programs to find how many visits for each
    target have been used, and in which ways.
 
    Returns an astropy table with the summary"""
 
 
    wf_pids = [2586, 2724, 2725, 2726]

    tables = dict()
    for pid in wf_pids:
        tables[pid] = query_program_status_form(pid)
        if verbose:
            print(pid)
            summarize_status(tables[pid])


    # Merged list of all targets, and all available status codes
    target_list = list(np.concatenate([tables[pid]['Target(s)'] for pid in tables]))

    statcodes = ['Archived', 'Executed', 'Scheduled', 'Scheduling', 'Skipped', 'Withdrawn']


    unique_targets = list(set(target_list))
    unique_targets.sort()
 
    # Scan through all those tables and extract visit status into a nested dict
    targ_summary_dict = { targ: {s:0 for s in statcodes} for targ in unique_targets}
    for table in tables.values():
        for i in range(len(table)):
            targ = table.loc[i,'Target(s)']
            stat = table.loc[i,'Status']
            #print(targ, stat, targ_summary_dict[targ][stat])
            targ_summary_dict[targ][stat]+=1

    # Then extract from the nested dicts into an astropy table for convenience and better display
    data = [unique_targets] + [[targ_summary_dict[targ][s] for targ in unique_targets] for s in statcodes]
    colnames = ['Target(s)'] + statcodes
    target_summary = astropy.table.Table(data=data, names=colnames)

    return target_summary


def plot_used_wfsc_targets(target_summary):
    """Plot on the sky which WFS stars have been used"""

    used = 5 - target_summary['Scheduling'] - target_summary['Withdrawn']

    radecs = [(f"{t[6:8]} {t[8:10]} {int(t[10:14])/100} {t[14:17]} {t[17:19]} {int(t[19:23])/200}") for t in target_summary['Target(s)']]
    coords = c.SkyCoord(radecs, unit=(u.hourangle, u.deg))

    plt.figure(figsize=(16,9))
    ax = plt.gcf().add_subplot( projection='mollweide')
    plt.title(f"Cycle 1 WFSC Target Usage\n[Shown on ICRS RA, Dec. coords]\n")

    plt.xticks(ticks=np.radians([-120, -60, 0, 60, 120, 180]),
                   labels=['8$^h$', '4$^h$', '0$^h$', '20$^h$', '16$^h$', '12$^h$'], alpha=0.7)
    ax.grid(True, alpha=0.5)



    plot_x = coords.ra.wrap_at('180d').radian
    plot_y = coords.dec.radian
    # note we MUST FLIP THE SIGN FOR X, since matplotlib map projections
    # don't let us make R.A. increase to the left
    ax.scatter(-plot_x, plot_y , s=used*20, c=used,
               edgecolors='gray', lw=0.5,
               cmap= matplotlib.cm.coolwarm, marker='o') #, linestyle='none')


def dsn_schedule(return_table=False, verbose=True, lookback=0.5*u.day):
    """ Retrieve and display DSN contact schedule
    """

    import pandas
    import requests
    from io import StringIO, BytesIO
    import pytz
    import astropy.table, astropy.time, astropy.units as u

    # Retreive schedule from web, and parse to astropy table, via Pandas
    dsn_jwst_url = 'https://spsweb.fltops.jpl.nasa.gov/rest/ops/info/activity/JWST/'
    req = requests.get(dsn_jwst_url)
    tab = pandas.read_html(BytesIO(req.content))[0]   # Parse HTML to list of tables, then take the first and only table in that list
    dsn_table = astropy.table.Table.from_pandas(tab)


    # Convert beginning & end of transit times into astropy Time
    dsn_table['BOT'] = astropy.time.Time(dsn_table['BOT'])
    dsn_table['EOT'] = astropy.time.Time(dsn_table['EOT'])

    # Find comm passes today-ish (between previous UTC midnight, and tomorrow UTC noon)
    now = astropy.time.Time.now()
    tstart, tend = astropy.time.Time(np.floor(now.mjd), format='mjd'), astropy.time.Time(np.ceil(now.mjd), format='mjd')
    tstart -= lookback
    todays_dsn = (dsn_table['BOT'] > tstart) & (dsn_table['EOT'] < (tend+12*u.hour))

    # now some timezone math
    tz = pytz.timezone('US/Eastern')
    dsn_table['BOT_Baltimore'] = [d.isoformat() for d in dsn_table['BOT'].to_datetime(tz)]
    dsn_table['EOT_Baltimore'] = [d.isoformat() for d in dsn_table['EOT'].to_datetime(tz)]

    # And print
    if verbose:
        print(" ")
        print(f"      \t Location\tContact Time Period [UTC]        \tContact Time [Baltimore, US/Eastern]\tActivity")
        print(f"      \t---------\t------------------------------------\t------------------------------------\t--------")
        shown_now = False
        for i, row in enumerate(dsn_table[todays_dsn]):
            ni = np.clip(i+1, 0, sum(todays_dsn)-1)
            #print(i, ni)
            next_row = dsn_table[todays_dsn][ni]
            ant = row['FACILITY']
            if ant < 30:
                loc = 'Goldstone'
            elif ant < 50:
                loc = ' Canberra'
            else:
                loc = '   Madrid'

            print(f"DSN-{ant}\t{loc}\t{row['BOT'].iso[0:16]} to {row['EOT'].iso[0:16]}\t{row['BOT_Baltimore'][0:16]} to {row['EOT_Baltimore'][0:16]}\t{row['ACTIVITY']}")
            if not shown_now:
                if (row['BOT']< now) and (row['EOT'] > now):
                    print(f"\t\t>> NOW: {now.iso[0:16]}\t\t\tDuring pass. Time remaining in contact: {(row['EOT']-now).to(u.hour):.2f}")
                    shown_now = True

                elif (row['BOT']< now) and (next_row['BOT'] > now):
                    print(f"\t\t>> NOW: {now.iso[0:16]}\t\t\tBetween passes. Time to next contact: {(next_row['BOT']-now).to(u.hour):.2f}")
                    shown_now = True

        print("\nThe above is based on the *planned* DSN schedule. Actual contact times may vary due to operational circumstances.\n")

    if return_table:
        return dsn_table[todays_dsn]

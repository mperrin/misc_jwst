import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from misc_jwst.command_line import *
from misc_jwst.command_line import _short_modes
import misc_jwst.visit_status
import collections
import textwrap
import pytz


def setup_plot(ax=None, tstart=None, tend=None):
    """ Configure axes for the schedule status plot"""
    now = astropy.time.Time.now()
    if tstart is None:
        tstart = now - 1*u.day
    if tend is None:
        tend = now + 1*u.day
    if ax is None:
        fig, axes = plt.subplots(figsize=(16,9), nrows=3, gridspec_kw={'height_ratios': [2,2,1], 'hspace':0})
    for ax in axes:
        ax.xaxis.axis_date()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        ax.set_xlim(tstart.plot_date, tend.plot_date)
        ax.set_ylim(0,1)
        ax.axvline(now.plot_date, ls='--', color='green')
        ax.set_yticks([])

    axes[0].set_ylabel("Scheduled Visits", fontsize='large', fontweight='bold')
    axes[0].set_yticks([0.33, 0.66])
    axes[0].set_yticklabels(['Prime', 'Parallel'])

    axes[1].set_yticks([0.33, 0.66])
    axes[1].set_yticklabels(['Prime', 'Parallel'])

    axes[1].set_ylabel("Actual Visits", fontsize='large', fontweight='bold')
    axes[2].set_ylabel("Engineering\n and\n Communications\n", fontsize='large', fontweight='bold')
    axes[2].set_xlabel("Time [UTC]", fontweight='bold')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\nDay of year %j'))
    axes[2].text(now.plot_date+0.005, 0.1, "Current Time Now", color='green', rotation=90)

    return fig, axes


def duration_to_timedelta(duration):
    """Parse the specific time format in the schedule table"""
    days, hours = duration.split('/')
    h, m,s = hours.split(':')
    return astropy.time.TimeDelta(int(days)*u.day + int(h)*u.hour + int(m)*u.minute + int(s)*u.second)


def draw_box(tstart, tend, ypos, height, color='white', edgecolor='black', text="", ax=None, fill_alpha=0.5, fontsize='x-small', text_offset = 1,
             set_clip_path=True, time_range_start=None, time_range_end=None,
             **kwargs):
    """Draw a box on the schedule status plot, with text annotation
    Bunch of extra complications for cosmetic display of text that attempts to minimizes overlap
    """
    if ax is None:
        ax = plt.gca()

    # Clip the size of the box to the displayed time range.
    # This is cosmetic, to improve display of the text labels and clipping at plot edges.
    # Works around matplotlib issue in which the patches themselves clip at plot axes but
    # text that uses the patch as a clip path does not.
    box_start_time = max(tstart.plot_date, time_range_start.plot_date if time_range_start else 0)
    box_end_time = min(tend.plot_date, time_range_end.plot_date if time_range_end else np.inf)

    plot_time_range = time_range_end.plot_date - time_range_start.plot_date if time_range_end else 1.5

    # Draw the box
    box = matplotlib.patches.Rectangle((box_start_time, ypos-height/2), (box_end_time-box_start_time), height,
                                       facecolor=matplotlib.colors.to_rgba(color, fill_alpha),
                                       edgecolor=edgecolor,
                                       **kwargs)
    ax.add_patch(box)

    # Draw the text. By default in the middle, but if it's a short visit try to
    # move them around in order to avoid text from adjacent boxes overlapping each other.
    # This is a bit ad hoc and imperfect but better than nothing.
    is_brief_visit = (tend.plot_date-tstart.plot_date) < 0.05 * plot_time_range   # TBD threshold
    horizontalalignment = 'left' if is_brief_visit else 'center'
    clip_on = False if is_brief_visit else True
    text_xpos = tstart.plot_date+0.005 if is_brief_visit else (tstart.plot_date + tend.plot_date)/2
    text_ypos = ypos - (text_offset-1)*height/3.2 * is_brief_visit

    xlim = ax.get_xlim()
    if xlim[0] < text_xpos < xlim[1]:

        text = ax.text(text_xpos, text_ypos, text, color=edgecolor,
        #ax.text(tstart.plot_date, text_ypos, text,
                verticalalignment='center', horizontalalignment=horizontalalignment,
                fontsize=fontsize, clip_on=clip_on, **kwargs)
        if set_clip_path:
            text.set_clip_path(box)


def draw_dsn_box(row, axes):
    """Illustrate DSN communications pass, as a shaded box behind the data"""
    ant = row['FACILITY']
    # Antenna number tells which site.
    if ant < 30:
        loc = 'Goldstone'
        textcolor='darkgoldenrod'
        color='yellow'
    elif ant < 50:
        loc = 'Canberra'
        textcolor = 'C0'
        color = 'C0'
    else:
        loc = 'Madrid'
        textcolor='forestgreen'
        color='C2'

    axes.fill_betweenx([0,1], row['BOT'].plot_date, row['EOT'].plot_date,
                     color=color, alpha=0.2, zorder=-5)
    axes.text((row['BOT'].plot_date+row['EOT'].plot_date)/2, 0.8, f"{loc}\n{ant}", color=textcolor,
             horizontalalignment='center', fontsize='medium', clip_on=True)


def get_visit_info(schedule_row):
    """Return various things we will plot.
    Simple helper function for minor code reuse to minimize repetition"""
    long_mode = schedule_row['SCIENCE INSTRUMENT AND MODE']
    short_mode = _short_modes.get(str(long_mode), str(long_mode))
    targname = schedule_row['TARGET NAME'] if schedule_row['TARGET NAME'] else "N/A"
    try:
        color = '0.3' if 'Dark' in long_mode else _colors[long_mode.split()[0]]
    except (KeyError, AttributeError):
        color = '0.5'
    return long_mode, short_mode, targname, color



# Define lookup table of colors for plotting visits
_colors = {'NIRCam':  'lightyellow',
           'WFSC':    'gold',
           'NIRISS':  'palegreen',
           'NIRSpec': 'skyblue',
           'MIRI': 'salmon',
           'FGS': 'lightgray',
           'Unknown': 'lightgray',
           'Realtime': 'lightgray',
           'Station': 'purple'}


def schedule_plot(trange = 1*u.day, open_plot=True, verbose=True, future=False):
    now = astropy.time.Time.now()
    # start and end times for the plot
    if future:
        # show mostly the future
        tstart = now - 0.5*trange
        tend = now + trange

    else:
        # show mostly the past
        tstart = now - trange
        tend = now + 0.5*trange
    # start time for retriving info should be well before start time for plot, to get any ongoing visits at that time.
    start_time = now - trange * 2


    #------------------- Retrieve information ----------------------
    # Retrieve information from various sources on the web or in MAST
    # Log of executed visits
    if verbose:
        print("Retrieving status...")
    log = engdb.get_ictm_event_log(startdate=start_time)
    # Retrieve DSN comm schedule from the web
    dsn_table = misc_jwst.visit_status.dsn_schedule(verbose=False, return_table=True, lookback=trange)
    # Retrieve schedule from the web
    schedule_full = get_schedule_table()

    #------------------- Parse that retrieved information and prepare tables, lists, etc to plot ----------------------
    latest_log_time = astropy.time.Time(log[-1]['Time'])
    visit_table = engdb.visit_start_end_times(log, verbose=False, return_table=True)
    visit_table['VISIT ID'] = [f"{int(v[1:6])}:{int(v[6:9])}:{int(v[9:12])}" for v in list(visit_table['visitid'].value)]

    # Find prime visits in schedule, by ignoring parallels which don't have a scheduled time
    has_dates = [str(v).startswith("20") for v in schedule_full['SCHEDULED START TIME'].value]
    schedule = schedule_full[has_dates]

    # Parse log of observed visits, and use schedule to infer visit types and modes
    for row in visit_table:
        try:
            match_index = np.where(schedule['VISIT ID'] == row['VISIT ID'])[0][0]
            schedrow = schedule[match_index]
            sched_start_time = astropy.time.Time(schedrow['SCHEDULED START TIME'])
            long_mode = schedrow['SCIENCE INSTRUMENT AND MODE']
            mode = _short_modes.get(str(long_mode), str(long_mode))
            delta_time = row['visit_fgs_start'] - sched_start_time
        except IndexError:
            pass

    # Make lookup table of parallel visits from the schedule
    attached_parallels = collections.defaultdict(lambda : [])

    prior_prime_visit = None  # Keep track of the previous one, for use with slew parallels in particular
    for i, row in enumerate(schedule_full):
        if 'PRIME' in row['VISIT TYPE']:
            prime_visit = row['VISIT ID']
        elif row['VISIT TYPE'] == "COORDINATED PARALLEL":
            # coordinated parallel! Same visit num as prime
            attached_parallels[prime_visit].append(('coordinated', prime_visit, i)) # parallel type, attached parallel visit id, and index into this table
        elif row['VISIT TYPE'] == "PARALLEL PURE":
            attached_parallels[prime_visit].append(('pure', row['VISIT ID'], i))  # parallel type, attached parallel visit id, and index into this table
        elif row['VISIT TYPE'] == "PARALLEL SLEW CALIBRATION":
            attached_parallels[prime_visit].append(('slew', row['VISIT ID'], i))  # parallel type, attached parallel visit id, and index into this table
        else:
            # There can be extra rows with no visit type; these list extra targets, e.g. TA ref stars, for certain NIRSpec visits.
            # We can ignore these.
            pass


    # The schedule doesn't list end times but does list durations.
    # Use the durations to compute expected end times.
    # Update schedule table with start and end as astropy Times

    durations_list = [duration_to_timedelta(d) for d in schedule['DURATION']]        # list of individual datetimes
    durations = astropy.time.TimeDelta([d.jd for d in durations_list], format='jd')  # one datetime with an ndarray of values

    schedule['start_time'] = astropy.time.Time(schedule['SCHEDULED START TIME'])
    schedule['end_time'] = astropy.time.Time(schedule['SCHEDULED START TIME']) + durations


    ###--------------------  Plot the plot !-----------------------------------------
    if verbose:
        print("Plotting visit status...")
    fig, axes = setup_plot(tstart=tstart, tend=tend)

    ### Draw boxes for the scheduled visits

    in_time_range = (((tstart < schedule['start_time']) & (schedule['start_time'] < tend) ) |
                     ((tstart < schedule['end_time']) & (schedule['end_time'] < tend) ) )
    text_offsets = {}
    prev_visit_end_time = now - trange  # placeholder; this variable will be used for slew parallels display only
    for i, row in enumerate(schedule[in_time_range]):
        long_mode, mode, targname, color = get_visit_info(row)
        text_offsets[row['VISIT ID']] = np.mod(i,3)  # save same offsets to reuse below for consistency

        draw_box(row['start_time'], row['end_time'], 0.25, 0.4, ax=axes[0],
                color = color,
                text = mode + "\n" + row['VISIT ID'] + "\n" + targname,
                text_offset = text_offsets[row['VISIT ID']], time_range_start=tstart, time_range_end=tend)

        nparallels = len(attached_parallels[row['VISIT ID']])
        if nparallels > 0:
            for i_parallel, (parallel_type, parallel_visit, parallel_index) in  enumerate(attached_parallels[row['VISIT ID']]):
                pheight = 0.22 if nparallels < 3 else (0.15 if nparallels ==3 else 0.5/nparallels)
                poffset = 0 if nparallels < 3 else 0.04  if nparallels ==3 else 0.1
                p_long_mode, p_mode, p_targ,  p_color = get_visit_info(schedule_full[parallel_index])

                if parallel_type == 'slew':
                    pstart, pend = prev_visit_end_time, row['start_time']
                else:
                    pstart, pend = row['start_time'], row['end_time']


                draw_box(pstart, pend, 0.6 + i_parallel*(pheight) - poffset, pheight, ax=axes[0],
                        color = p_color,
                        text = p_mode + "\n" + parallel_visit,
                        text_offset = text_offsets[row['VISIT ID']], time_range_start=tstart, time_range_end=tend)

        prev_visit_end_time = row['end_time']



    ###---------------------------------------------------------------
    ### Draw boxes for the actual visits

    in_time_range = (((tstart < visit_table['visitstart']) & (visit_table['visitstart'] < tend) ) |
                     ((tstart < visit_table['visitend']) & (visit_table['visitend'] < tend) ) )
    prev_visit_end_time = now - trange  # similar to above, will be used for slew parallels display
    for i, row in enumerate(visit_table[in_time_range]):
        try:
            match_index = np.where(schedule['VISIT ID'] == row['VISIT ID'])[0][0]
            schedrow = schedule[match_index]
            long_mode, mode, targname, color = get_visit_info(schedrow)

            sched_start_time = astropy.time.Time(schedrow['SCHEDULED START TIME'])
            delta_time = row['visit_fgs_start'] - sched_start_time
            edgecolor = 'red' if (('SKIP' in row['notes']) or ('failed' in row['notes']) or ('FAILED' in row['notes'])  or
                ('unavailable' in row['notes']) or ('ERROR' in row['notes'])) else 'black'

            if i==len(visit_table[in_time_range])-1:
                #print(f"latest visit ΔT: {delta_time.to_value(u.hour):+.2f} hr")
                axes[1].text(row['visit_fgs_start'].plot_date, 0.85, f"  Latest visit ΔT:\n  {delta_time.to_value(u.hour):+.2f} hr", color='blue')
                for ax in axes:
                    ax.axvline(row['visit_fgs_start'].plot_date, ls=':', color='blue')
            
        except IndexError:
            # Handle case where there is no match in the schedule, e.g. due to an intercept, and the published schedule is not yet updated.
            long_mode, mode, targname, color = "Unknown", "", "", '0.7'
            edgecolor='black'

        if row['notes']:
            notes = row['notes'].replace("Still ongoing at end of available log.", "") # This message isn't needed, so don't display
            notes = "\n".join(textwrap.wrap(notes, width=25))
            targname += "\n" + notes
        try:
            color = '0.3' if 'Dark' in long_mode else _colors[long_mode.split()[0]]
        except (KeyError, AttributeError):
            color = '0.5'
        draw_box(row['visit_fgs_start'], row['visitend'], 0.25, 0.4, ax=axes[1],
                color = color, edgecolor=edgecolor,
                text = mode + "\n" + row['VISIT ID'] + "\n" + targname, fontsize='x-small',
                text_offset = text_offsets.get(row['VISIT ID'], 0), time_range_start=tstart, time_range_end=tend)

        if (nparallels := len(attached_parallels[row['VISIT ID']])) > 0:

            pheight = 0.22 if nparallels < 3 else (0.15 if nparallels ==3 else 0.5/nparallels)
            poffset = 0 if nparallels < 3 else 0.04  if nparallels ==3 else 0.1

            for i_parallel, (parallel_type, parallel_visit, parallel_index) in  enumerate(attached_parallels[row['VISIT ID']]):
                if parallel_type == 'slew':
                    pstart, pend = prev_visit_end_time, row['visit_fgs_start']
                else:
                    pstart, pend = row['visit_fgs_start'], row['visitend']

                p_long_mode, p_mode, p_targ,  p_color = get_visit_info(schedule_full[parallel_index])
                draw_box(pstart, pend, 0.6 + i_parallel*pheight - poffset, pheight, ax=axes[1],
                        color = p_color, edgecolor=edgecolor,
                        text = p_mode + "\n" + parallel_visit,
                        text_offset = text_offsets.get(row['VISIT ID'],0), time_range_start=tstart, time_range_end=tend)
        prev_visit_end_time = row['visitend']

    axes[1].fill_between([latest_log_time.plot_date, tend.plot_date], 0, 1, color='0.95', zorder=-10)
    axes[1].axvline(latest_log_time.plot_date, color='0.5', lw=0.5)
    axes[1].text((latest_log_time+0.5*u.hour).plot_date, 0.5, "Observatory logs\nnot yet available", verticalalignment='center',  horizontalalignment='left', fontweight='light', fontsize='small')

    oplabel = schedule_full.meta['op_packages'][1 if tstart > schedule_full.meta['week2_start_time'] else 0]
    axes[0].text(0.005, 0.99, f"OP package: {oplabel}", transform=axes[0].transAxes, verticalalignment='top', fontsize='small')
    # If the boundary between scedules is in the time range shown, display it
    if (tstart < schedule_full.meta['week2_start_time']) & (schedule_full.meta['week2_start_time'] < tend):
        week2_start_time = astropy.time.Time(schedule_full.meta['week2_start_time'])
        oplabel2 = schedule_full.meta['op_packages'][1 ]
        axes[0].text(week2_start_time.plot_date, 0.99, f"  OP package: {oplabel2}",  verticalalignment='top', fontsize='small')
        axes[0].axvline(week2_start_time.plot_date, color='black', ls='-', lw=0.5)

    # There may be some gaps where we do not yet have log messages available in the MAST EngDB
    # let's also mark those in gray
    # Find timesteps between successive log messages, in floating point days
    delta_times = log[1:]['MJD'] - log[:-1]['MJD']
    # find spaces where there is more than 1 hour between log messages
    maybe_gaps = np.where(delta_times > 1/24)
    # Some of those may be exposures > 1 hour in duration
    # we can check this by looking at the subsequent log message after each gap
    gap_isnt_long_exposure = ['completed exptime' not  in msg for msg in log[maybe_gaps[0]+1]['Message']]
    # actual gaps are any that aren't long exposures
    actual_gaps = maybe_gaps[0][gap_isnt_long_exposure]
    for i in actual_gaps:
        t0 = astropy.time.Time(log[i  ]['Time'])
        t1 = astropy.time.Time(log[i+1]['Time'])
        dt = t1-t0
        axes[1].fill_between([t0.plot_date,
                              t1.plot_date], 0, 1, color='0.95', zorder=-10)
        axes[1].axvline(t0.plot_date, color='0.5', lw=0.5)
        axes[1].axvline(t1.plot_date, color='0.5', lw=0.5)
        axes[1].text((t0+dt/2).plot_date, 0.75, "Observatory logs\nnot yet available,\nor observatory was idle", verticalalignment='center',  horizontalalignment='center', fontweight='light', fontsize='small')

    ###---------------------------------------------------------------
    ### Engineering Metadata
    for row in dsn_table:
        draw_dsn_box(row, axes[2])

    if verbose:
        print("Retrieving and plotting engineering quantities...")

    # TODO: 
    # - sun pitch, sun roll
    # - data volume onboard
    table_ssr = misc_jwst.engdb.get_mnemonic('SF_ZSSRFREE', startdate=tstart.isot, enddate=tend.isot)
    table_ssr['Time'] = astropy.time.Time(table_ssr['Time'])
    ssr_size =  62093134
    table_ssr['SSR_FRAC_USED'] = (ssr_size - table_ssr['SF_ZSSRFREE'])/ssr_size
    axes[2].plot(table_ssr['Time'].plot_date, table_ssr['SSR_FRAC_USED'], color='Magenta', label='SSR Storage Used')

    # - momentum stored
    table_momentum = misc_jwst.engdb.get_mnemonic('SA_ZMOMMAG', startdate=tstart.isot, enddate=tend.isot)
    table_momentum['Time'] = astropy.time.Time(table_momentum['Time'])
    mom_max = 90 # TBD
    plt.plot(table_momentum['Time'].plot_date, table_momentum['SA_ZMOMMAG']/mom_max, color='teal', label='Stored Momentum')

    axes[2].legend(loc='upper right')

    # - J frame? 

    for ax in axes:
        ax.set_xlim(tstart.plot_date, tend.plot_date)

    tz = pytz.timezone('US/Eastern')
    fig.text(0.01, 0.01, f"Updated:       {now.iso[0:16]} UTC       {now.to_datetime(tz).isoformat()[0:16]} Baltimore",
             fontsize='small')
    plt.tight_layout()
    plt.savefig('current_timeline_plot.png')
    if open_plot:
        import os
        os.system("open current_timeline_plot.png")
    if verbose:
        print("Saved to current_timeline_plot.png")




import os, sys
import requests
import functools
from astroquery.mast import Mast,Observations
from astropy.table import Table, unique, vstack
import astropy.time
import astropy.io.fits as fits
import numpy as np
import scipy

import matplotlib, matplotlib.pyplot as plt

import pysiaf
import jwst.datamodels

def mast_retrieve_guiding_files(filenames, out_dir='.', verbose=True):
    """Download one or more guiding data products from MAST

    If the file is already present in the specified local directory, it's not downloaded again.

    """

    mast_url='https://mast.stsci.edu/api/v0.1/Download/file'
    uri_prefix = 'mast:JWST/product/'

    outputs = []

    mast_api_token = os.environ.get('MAST_API_TOKEN')
    if mast_api_token is not None:
        headers=dict(Authorization=f"token {mast_api_token}")
    else:
        headers=None

    for p in filenames:
        outfile = os.path.join(out_dir, p)

        if os.path.isfile(outfile):
            if verbose:
                print("ALREADY DOWNLOADED: ", outfile)
            outputs.append(outfile)
            continue

        r = requests.get(mast_url, params=dict(uri=uri_prefix+p), stream=True,
                    # include the following argument if authentication is needed
                    #headers=dict(Authorization=f"token {mast_api_token}"))
                         headers=headers,
                        )
        r.raise_for_status()
        with open(outfile, 'wb') as fd:
            for data in r.iter_content(chunk_size=1024000):
                fd.write(data)

        if not os.path.isfile(outfile):
            if verbose:
                print("ERROR: " + outfile + " failed to download.")
        else:
            if verbose:
                print("COMPLETE: ", outfile)
            outputs.append(outfile)
    return outputs

def set_params(parameters):
    """Utility function for making dicts used in MAST queries"""
    return [{"paramName":p, "values":v} for p,v in parameters.items()]


@functools.lru_cache
def find_relevant_guiding_file(sci_filename, verbose=True):
    """ Given a filename of a JWST science file, retrieve the relevant guiding data product.
    This uses FITS keywords in the science header to determine the time period and guide mode,
    and then retrieves the file from MAST

    """


    sci_hdul = fits.open(sci_filename)

    progid = sci_hdul[0].header['PROGRAM']
    obs = sci_hdul[0].header['OBSERVTN']
    guidemode = sci_hdul[0].header['PCS_MODE']


    # Set up the query
    keywords = {
    'program': [progid]
    ,'observtn': [obs]
    ,'exp_type': ['FGS_'+guidemode]
    }

    params = {
        'columns': '*',
        'filters': set_params(keywords)
        }


    # Run the web service query. This uses the specialized, lower-level webservice for the
    # guidestar queries: https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar

    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)


    if len(t) > 0:
        # Ensure unique file names, should any be repeated over multiple observations (e.g. if parallels):
        fn = list(set(t['fileName']))
        # Set of derived Observation IDs:

        products = list(set(fn))
        # If you want the uncals instead do this:
        #products = list(set([x.replace('_cal','_uncal') for x in fn]))
    products.sort()


    if verbose:
        print(f"For science data file: {sci_filename}")
        print("Found guiding telemetry files:")
        for p in products:
            print("   ", p)

    # Some guide files are split into multiple segments, which we have to deal with.
    guide_timestamp_parts = [fn.split('_')[2] for fn in products]
    is_segmented = ['seg' in part for part in guide_timestamp_parts]
    for i in range(len(guide_timestamp_parts)):
        if is_segmented[i]:
            guide_timestamp_parts[i] = guide_timestamp_parts[i].split('-')[0]
    guide_timestamps = np.asarray(guide_timestamp_parts, int)
    t_beg = astropy.time.Time(sci_hdul[0].header['DATE-BEG'])
    t_end = astropy.time.Time(sci_hdul[0].header['DATE-END'])
    obs_end_time = int(t_end.strftime('%Y%j%H%M%S'))

    delta_times = np.array(guide_timestamps-obs_end_time, float)
    # want to find the minimum delta which is at least positive
    #print(delta_times)
    delta_times[delta_times<0] = np.nan
    #print(delta_times)

    wmatch = np.argmin(np.abs(delta_times))
    wmatch = np.where(delta_times ==np.nanmin(delta_times))[0][0]
    delta_min = (guide_timestamps-obs_end_time)[wmatch]

    if verbose:
        print("Based on science DATE-END keyword and guiding timestamps, the matching GS file is: ")
        print("   ", products[wmatch])
        print(f"    t_end = {obs_end_time}\t delta = {delta_min}")

    if is_segmented[wmatch]:
        # We ought to fetch all the segmented GS files for that guide period
        products_to_fetch = [fn for fn in products if fn.startswith(products[wmatch][0:33])]
        if verbose:
            print("   That GS data is divided into multiple segment files:")
            print("   ".join(products_to_fetch))
    else:
        products_to_fetch = [products[wmatch],]

    outfiles = mast_retrieve_guiding_files(products_to_fetch)

    return outfiles


@functools.lru_cache
def find_visit_guiding_files(visitid, guidemode='FINEGUIDE', verbose=True, autodownload=True):
    """ Given a JWST visit id string, like 'V01234005001', retrieve all guiding data products
    for that visit from MAST. Files downloaded into current working directory.

    """

    progid = visitid[1:6]
    obs = visitid[6:9]

    if guidemode.upper()=='ID':
        exp_type = 'FGS_ID-IMAGE'
    elif guidemode.upper()=='ACQ' or guidemode.upper()=='ACQ1':
        exp_type = 'FGS_ACQ1'
    elif guidemode.upper()=='ACQ2':
        exp_type = 'FGS_ACQ2'
    elif guidemode.upper() == 'TRACK':
        exp_type = 'FGS_TRACK'
    elif guidemode.upper() == 'FINEGUIDE':
        exp_type = 'FGS_FINEGUIDE'
    else:
        raise ValueError(f"Unknown/invalid guidemode parameter: {guidemode}")


    # Set up the query
    keywords = {
    'program': [progid]
    ,'observtn': [obs]
    ,'exp_type': [exp_type]
    }

    params = {
        'columns': '*',
        'filters': set_params(keywords)
        }

    # Run the web service query. This uses the specialized, lower-level webservice for the
    # guidestar queries: https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar

    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)


    if len(t) > 0:
        # Ensure unique file names, should any be repeated over multiple observations (e.g. if parallels):
        fn = list(set(t['fileName']))
        # Set of derived Observation IDs:

        products = list(set(fn))
        # If you want the uncals instead do this:
        #products = list(set([x.replace('_cal','_uncal') for x in fn]))
        products.sort()
    else:
        print("Query returned no guiding files")
        return None

    if verbose:
        print(f"For visit: {visitid}")
        print("Found guiding telemetry files:")
        for p in products:
            print("   ", p)

    if autodownload:
        outfiles = mast_retrieve_guiding_files(products)
        return outfiles
    else:
        return products



@functools.lru_cache
def find_guiding_id_file(sci_filename=None, guidemode='ID', progid=None, obs=None, visit=1, visitid=None, verbose=True,
        autodownload=True):
    """ Given a filename of a JWST science file, retrieve the relevant guiding ID data product.
    (or Acq data product)
    This uses FITS keywords in the science header to determine the time period and guide mode,
    and then retrieves the file from MAST

    """

    if sci_filename is not None:
        sci_hdul = fits.open(sci_filename)
        progid_str = sci_hdul[0].header['PROGRAM']
        obs_str = sci_hdul[0].header['OBSERVTN']
        visit_str = sci_hdul[0].header['VISIT_ID']
    elif visitid is not None:
        progid_str = visitid[1:6]
        obs_str = visitid[6:9]
        visit_str = visitid[1:12]
    elif progid is None or obs is None or visit is None:
        raise RuntimeError("You must supply either the sci_filename parameter, or visitid, or both progid and obs (optionally also visit)")
    else:
        progid_str = f'{progid:05d}'
        obs_str = f'{obs:03d}'
        visit_str = f'{progid:05d}{obs:03d}{visit:03d}'

    return find_visit_guiding_files(visitid='V'+visit_str, guidemode='ID')



def guiding_performance_plot(sci_filename=None, visitid=None, verbose=True, save=False, yrange=None,
                             time_range_fraction=None):
    """Generate a plot showing the guiding jitter during an exposure or visit


    """
    if sci_filename is None and visitid is None:
        raise RuntimeError("You must set either sci_filename or visitid")

    # Retrieve the guiding packet file(s) from MAST
    if visitid:
        visit_mode = True
        gs_fns = find_visit_guiding_files(visitid, verbose=verbose)
    else:
        visit_mode = False
        gs_fns = find_relevant_guiding_file(sci_filename)

        # Determine start and end times for the exposure
        with fits.open(sci_filename) as sci_hdul:
            t_beg = astropy.time.Time(sci_hdul[0].header['DATE-BEG'])
            t_end = astropy.time.Time(sci_hdul[0].header['DATE-END'])

    gs_fn = gs_fns[0]

    # We may have multiple GS filenames returned, in the case where the data is split into several segments
    # If so, concatenat them

    dither_times = []
    last_gs_fn_middle= ''
    for i, gs_fn in enumerate(gs_fns):
        gs_fn_base = os.path.splitext(os.path.basename(gs_fn))[0]

        if i==0:  # For a single file or first segment, just read it in
            # Read the data from that file, and parse into astropy Times
            pointing_table = astropy.table.Table.read(gs_fn, hdu=4)
            centroid_table = astropy.table.Table.read(gs_fn, hdu=5)
            display_gs_fn = os.path.basename(gs_fn)
            if visit_mode:
                # We have to compute means per segment, since it's not meaningful to combine across dithers
                mask = centroid_table.columns['bad_centroid_dq_flag'] == 'GOOD'
                xmean = np.nanmean(centroid_table[mask]['guide_star_position_x'])
                ymean = np.nanmean(centroid_table[mask]['guide_star_position_y'])
                centroid_table['guide_star_position_x'] -= xmean
                centroid_table['guide_star_position_y'] -= ymean


        else:  # for a later segment, read in and append
            pointing_table_more = astropy.table.Table.read(gs_fn, hdu=4)
            centroid_table_more = astropy.table.Table.read(gs_fn, hdu=5)
            # mark one point as NaN, to visually show a gap in the plot
            centroid_table_more['guide_star_position_x'][-1] = np.nan
            centroid_table_more['guide_star_position_y'][-1] = np.nan
            centroid_table_more['bad_centroid_dq_flag'][-1] = 'GOOD'

            if visit_mode:
                fn_middle = os.path.basename(gs_fn).split('-')[1]
                if fn_middle != last_gs_fn_middle: # New time in file stamp, after a dither and reacq
                    dither_times.append(astropy.time.Time(pointing_table_more.columns['time'][0], format='mjd') )
                    last_gs_fn_middle = fn_middle
                    if verbose:
                        print(f"Dither before {gs_fn} at {dither_times[-1].iso}")
                # We have to compute means per segment, since it's not meaningful to combine across dithers
                mask = centroid_table_more.columns['bad_centroid_dq_flag'] == 'GOOD'
                xmean = np.nanmean(centroid_table_more[mask]['guide_star_position_x'])
                ymean = np.nanmean(centroid_table_more[mask]['guide_star_position_y'])
                centroid_table_more['guide_star_position_x'] -= xmean
                centroid_table_more['guide_star_position_y'] -= ymean

            pointing_table = astropy.table.vstack([pointing_table, pointing_table_more], metadata_conflicts='silent')
            centroid_table = astropy.table.vstack([centroid_table, centroid_table_more], metadata_conflicts='silent')
            display_gs_fn = os.path.basename(gs_fn)[0:33]+ "-seg*_cal.fits"
    if visit_mode:
        display_gs_fn = f"All (n={len(gs_fns)}) guidestar files for {visitid}"

    mask = centroid_table.columns['bad_centroid_dq_flag'] == 'GOOD'

    ctimes = astropy.time.Time(centroid_table['observatory_time'])
    ptimes = astropy.time.Time(pointing_table.columns['time'], format='mjd')

    # Compute the mean X and Y positions
    xmean = np.nanmean(centroid_table[mask]['guide_star_position_x'])
    ymean = np.nanmean(centroid_table[mask]['guide_star_position_y'])


    # Create Plots
    fig, axes = plt.subplots(figsize=(16,12), nrows=3)

    min_time = np.min([ptimes.plot_date.min(), ctimes.plot_date.min()])
    max_time = np.max([ptimes.plot_date.max(), ctimes.plot_date.max()])
    dtime = max_time - min_time

    if not visit_mode:
        # Find the subset of centroid data from during that exposure
        ptimes_during_exposure = (t_beg < ptimes ) & (ptimes < t_end)
        ctimes_during_exposure = (t_beg < ctimes[mask] ) & (ctimes[mask] < t_end)


    for i in range(3):
        axes[i].xaxis.axis_date()
        axes[i].set_xlim(min_time -0.01*dtime, max_time+0.01*dtime)

    if visit_mode:
        axes[0].set_title(f"Guiding during {visitid}", fontweight='bold', fontsize=18)
        if len(dither_times)>0:
            axes[1].text(0.5, 0.05, "Centroids offsets shown are relative to the mean position within each guider file (usually per dither)",
                         transform=axes[1].transAxes, horizontalalignment='center')
    else:
        axes[0].set_title(f"Guiding during {os.path.basename(sci_filename)}", fontweight='bold', fontsize=18)

    axes[0].semilogy(ptimes.plot_date, pointing_table.columns['jitter'], alpha=0.9, color='C0')
    axes[0].set_ylim(1e-2, 1e3)
    axes[0].set_ylabel("Jitter\n[mas]", fontsize=18)
    axes[0].text(0.01, 0.95, display_gs_fn, fontsize=16, transform=axes[0].transAxes, verticalalignment='top')

    axes[0].text(ptimes.plot_date.min(), 100, " Guiding Start", color='C0', clip_on=True)
    axes[0].text(ptimes.plot_date.max(), 100, "Guiding End ", color='C0',
                horizontalalignment='right', clip_on = True)


    axes[1].plot(ctimes[mask].plot_date, centroid_table[mask]['guide_star_position_x']-xmean, label='X Centroids', color='C1', alpha=0.7)
    axes[1].plot(ctimes[mask].plot_date, centroid_table[mask]['guide_star_position_y']-ymean, label='Y Centroids', color='C4', alpha=0.7)
    axes[1].legend(loc='lower right')
    axes[1].set_ylabel("GS centroid offsets\n[arcsec]", fontsize=18)
    axes[1].axhline(0, ls=":", color='gray')
    if yrange is not None:
        axes[1].set_ylim(*yrange)

    axes[2].plot(ctimes.plot_date, mask, label='GOOD Centroids', color='C1')
    frac_good = mask.sum() / len(mask)
    print(f'Fraction of good centroids: {frac_good}')
    if frac_good < 0.95:
        axes[2].text(0.5, 0.9, f"WARNING, {(1-frac_good)*100:.2f}% of guider centroids were BAD during this.",
                     color='red', fontweight='bold', transform=axes[2].transAxes, horizontalalignment='center')

    if not visit_mode:
        axes[0].text(t_beg.plot_date, 20,
                 f"    mean jitter: {pointing_table.columns['jitter'][ptimes_during_exposure].mean():.2f} mas", color='green')
        axes[0].text(t_beg.plot_date, 50, " Exposure", color='green')

        axes[0].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
        axes[1].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
        axes[2].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
    else:
        for dithertime in dither_times:
           for ax in axes:
                ax.axvline(dithertime.plot_date, color='black', ls='--')
           axes[2].text(dithertime.plot_date, -0.35, 'SAM', rotation=90, clip_on=True)

        import misc_jwst.mast
        exposure_table = misc_jwst.mast.get_visit_exposure_times(visitid)
        already_plotted_exptimes = set()
        for row in exposure_table:
            if row['date_beg_mjd'].plot_date in already_plotted_exptimes:
                continue
            if verbose:
                print(f"Exposure {row['filename']} began at {row['date_beg_mjd'].iso}")

            for ax in axes:
                ax.axvspan(row['date_beg_mjd'].plot_date, row['date_end_mjd'].plot_date, color='green', alpha=0.15)
            axes[2].text(row['date_beg_mjd'].plot_date, -0.25, row['filename'], color='green', rotation=10, fontsize='small',
                         clip_on=True)
            already_plotted_exptimes.add(row['date_beg_mjd'].plot_date)


    axes[2].set_ylabel("Centroid Quality Flag\n", fontsize=18)
    axes[2].set_yticks((0,1))
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].set_yticklabels(['BAD', 'GOOD'])

    if time_range_fraction:
        tl = axes[0].get_xlim()
        t_duration = tl[1] - tl[0]
        for ax in axes:
            ax.set_xlim(tl[0] + time_range_fraction[0]*t_duration,
                        tl[0] + time_range_fraction[1]*t_duration)





    if visit_mode:
        outname = f'guidingplot_{visitid}.pdf'
    else:
        outname = f'guidingplot_{gs_fn_base}.pdf'

    if save:
        plt.savefig(outname)
        if verbose:
            print(f' ==> {outname}')



def guiding_performance_jitterball(sci_filename, fov_size = 8, nbins=50, verbose=True, save=False):
    """Generate a plot showing the guiding jitter during an exposure


    """

    # Retrieve the guiding packet file from MAST
    gs_fns = find_relevant_guiding_file(sci_filename)

    gs_fn = gs_fns[0]

    # Determine start and end times for the exposure
    with fits.open(sci_filename) as sci_hdul:
        t_beg = astropy.time.Time(sci_hdul[0].header['DATE-BEG'])
        t_end = astropy.time.Time(sci_hdul[0].header['DATE-END'])

    # We may have multiple GS filenames returned, in the case where the data is split into several segments
    # If so, concatenat them

    for i, gs_fn in enumerate(gs_fns):
        gs_fn_base = os.path.splitext(os.path.basename(gs_fn))[0]

        if i==0:  # For a single file or first segment, just read it in
            # Read the data from that file, and parse into astropy Times
            pointing_table = astropy.table.Table.read(gs_fn, hdu=4)
            centroid_table = astropy.table.Table.read(gs_fn, hdu=5)
            display_gs_fn = os.path.basename(gs_fn)
        else:  # for a later segment, read in and append
            pointing_table_more = astropy.table.Table.read(gs_fn, hdu=4)
            centroid_table_more = astropy.table.Table.read(gs_fn, hdu=5)
            pointing_table = astropy.table.vstack([pointing_table, pointing_table_more], metadata_conflicts='silent')
            centroid_table = astropy.table.vstack([centroid_table, centroid_table_more], metadata_conflicts='silent')
            display_gs_fn = os.path.basename(gs_fn)[0:33]+ "-seg*_cal.fits"


    mask = centroid_table.columns['bad_centroid_dq_flag'] == 'GOOD'

    ctimes = astropy.time.Time(centroid_table['observatory_time'])
    ptimes = astropy.time.Time(pointing_table.columns['time'], format='mjd')

    # Compute the mean X and Y positions
    xmean = centroid_table[mask]['guide_star_position_x'].mean()
    ymean = centroid_table[mask]['guide_star_position_y'].mean()


    # Create Plots
    fig= plt.figure(figsize=(8,8),)


    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    #scatter_hist(x, y, ax, ax_histx, ax_histy)



    min_time = np.min([ptimes.plot_date.min(), ctimes.plot_date.min()])
    max_time = np.max([ptimes.plot_date.max(), ctimes.plot_date.max()])
    dtime = max_time - min_time

    # Find the subset of centroid data from during that exposure
    ptimes_during_exposure = (t_beg < ptimes ) & (ptimes < t_end)
    ctimes_during_exposure = (t_beg < ctimes[mask] ) & (ctimes[mask] < t_end)

    xpos = centroid_table[mask][ctimes_during_exposure]['guide_star_position_x']
    ypos = centroid_table[mask][ctimes_during_exposure]['guide_star_position_y']

    xmean = xpos.mean()
    ymean = ypos.mean()

    rpos = np.sqrt(xpos**2+ypos**2)
    print(np.std(rpos))

    xoffsets = (xpos-xmean)*1000
    yoffsets = (ypos-ymean)*1000

    ax.scatter(xoffsets, yoffsets, alpha = 0.8, marker='+')
    ax.set_aspect('equal')

    ax.set_xlim(-fov_size/2, fov_size/2)
    ax.set_ylim(-fov_size/2, fov_size/2)
    ax.set_xlabel("GS Centroid Offset X [milliarcsec]", fontsize=18)
    ax.set_ylabel("GS Centroid Offset Y [milliarcsec]", fontsize=18)


    fig.suptitle(f"Guiding during {os.path.basename(sci_filename)}\n", fontweight='bold', fontsize=18)


    for rad in [1,2,3]:
        ax.add_artist(plt.Circle( (0,0), rad, fill=False, color='gray', ls='--'))
        if rad<fov_size/2:
            ax.text(0, rad+0.1, f"{rad} mas", color='gray')


    # Draw histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    bins = np.linspace(-fov_size/2, fov_size/2, nbins)
    ax_histx.hist(xoffsets, bins=bins)
    ax_histy.hist(yoffsets, bins=bins, orientation='horizontal')

    outname = f'guidingball_{gs_fn_base}.pdf'

    if save:
        plt.savefig(outname)
        if verbose:
            print(f' ==> {outname}')





import functools

@functools.cache
def get_siaf_aperture(guidername):
    """Get an FGS SIAF aperture
    With caching since loading SIAF is slow the first time"""
    gid = guidername[-1] # '1' or '2'
    s = pysiaf.Siaf('FGS')
    return s.apertures[f'FGS{gid}_FULL_OSS']  # not sure if the _OSS is needed in this case or not



@functools.cache
def get_visit_contents(visfilename):
    """Load a visit file, with caching since the
    same visit file may be used for several GS ID attempts
    """
    import visitviewer
    return visitviewer.VisitFileContents(visfilename)


def display_one_id_image(filename, destripe = True, smooth=True, ax=None,
                         show_metadata=True, plot_guidestars=True, count=0,
                        orientation='sci', return_model=False):
    """Display a JWST Guiding ID image

    Displays on a log stretch, optionally with overplotted guide star information if the visit files are available

    Parameters
    ----------
    destripe, smooth : Bool
        should we do some image processing to make the guide stars easier to see in the images

    orientation : 'sci' or 'raw':
        'sci' puts science frame 0,0 at lower left, like MAST output products seen in DS9
        'raw' puts detetor raw frame 0,0 at upper left, like the FGS DHAS tool's plots

    return_model : bool
        Optional, return the image model if you want to do some further image manipulations
        (for efficiency to avoid reloading it multiple times)
    """

    model = jwst.datamodels.open(filename)

    im = model.data[0]

    if ax is None:
        ax = plt.gca()

    if destripe:
        im = im - np.median(im, axis=0).reshape([1, im.shape[1]])

    if smooth:
        im = scipy.ndimage.median_filter(im, size=3)


    if orientation=='raw':
        # Display like in raw detector orientation, consistent with usage on-board and in FGS DHAS analyses

        if model.meta.instrument.detector=='GUIDER2':
            im = im.transpose()
            im = im[::-1]
        else:
            im = np.rot90(im)
            im = im[:, ::-1] # maybe??
        origin='upper' # Argh, the FGS DHAS diagnostic plots put 0,0 at the UPPER left
    else:
        origin='lower'

    mean, median, sigma = astropy.stats.sigma_clipped_stats(im, )

    norm = matplotlib.colors.AsinhNorm(vmin = median-sigma, vmax=10*sigma, linear_width=2*sigma)

    ax.imshow(im, norm=norm, origin=origin)

    ax.set_title(os.path.basename(filename))
    #ax.xaxis.set_ticks([])
    #ax.yaxis.set_ticks([])

    if show_metadata:
        ax.text(0.01, 0.99, f'{model.meta.instrument.detector}\nGS index: {model.meta.guidestar.gs_order}\n{model.meta.observation.date_beg[:-4]}',
                color='yellow',
                transform=ax.transAxes,
                verticalalignment='top')
        if orientation=='raw':
            ax.text(0.99, 0.99, f'detector raw orientation',
                color='yellow',
                transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right')


    if plot_guidestars:
        visfilename = f'V{model.meta.observation.visit_id}.vst'

        if os.path.exists(visfilename):
            if count==0: print(f'Found visit file {visfilename}')
            vis = get_visit_contents(visfilename)
            if count==0: print("Retrieving and plotting guide star info from visit file")

            gsinfo = vis.guide_activities[model.meta.guidestar.gs_order - 1]



            ap = get_siaf_aperture(model.meta.instrument.detector)
            pixscale = (ap.XSciScale + ap.YSciScale)/2

            if orientation=='raw':
                coord_transform = ap.idl_to_det
            else:
                def coord_transform(*args):
                    x,y = ap.idl_to_sci(*args)
                    return  y, model.data.shape[-1]-x  # I don't understand the coord transform and why this flip is needed but it works

            def add_circle(x,y, radius=10/pixscale/3, color='white',
                           label=None):
                ax.add_patch(matplotlib.patches.Circle( (x,y), radius, edgecolor=color,
                                                       facecolor='none'))
                ax.add_patch(matplotlib.patches.Circle( (x,y), radius*3, edgecolor=color,
                                                       facecolor='none', ls='dotted'))

                if label is not None:
                    ax.text(x, y, label, color=color, fontweight='bold', alpha=0.75)


            dety, detx = coord_transform(gsinfo.GSXID, gsinfo.GSYID)
            #ax.scatter(detx, dety, s=300, marker='o', color='orange', facecolor='none')

            add_circle(detx, dety, label=f'commanded\nGS candidate {model.meta.guidestar.gs_order }\n',
                       color='orange')

            for ref_id in range(1,10):
                if hasattr(gsinfo, f'REF{ref_id}X'):
                    dety, detx = coord_transform(getattr(gsinfo, f'REF{ref_id}X'),
                                               getattr(gsinfo, f'REF{ref_id}Y'))
                    add_circle(detx, dety,
                               label=f'commanded\nRef {ref_id}\n', color='magenta')

    if return_model:
        return model

def display_one_guider_image(filename,  ax=None, use_dq=False,
                         show_metadata=True, count=0,
                         orientation='sci', return_model=False):
    """Display a JWST Guiding Acq image

    Displays on a asinh stretch

    Parameters
    ----------
    orientation : 'sci' or 'raw':
        'sci' puts science frame 0,0 at lower left, like MAST output products seen in DS9
        'raw' puts detetor raw frame 0,0 at upper left, like the FGS DHAS tool's plots

    return_model : bool
        Optional, return the image model if you want to do some further image manipulations
        (for efficiency to avoid reloading it multiple times)
    """

    model = jwst.datamodels.open(filename)


    if model.data.ndim == 3 and 'track' in filename:
        # The star location in track images may move over time, by design.
        # So just take and display one slice here, rather than trying to average over many
        im = model.data[0]
    elif model.data.ndim == 3 and 'track' not in filename:
        # for fine guide images we can average over time, generally  (*small caveat for SGD subpix dithers)

        im = model.data.mean(axis=0)
    else:
        im = model.data
    dq = model.dq

    if ax is None:
        ax = plt.gca()

    if orientation=='raw':
        # Display like in raw detector orientation, consistent with usage on-board and in FGS DHAS analyses
        if model.meta.instrument.detector=='GUIDER2':
            im = im.transpose()
            im = im[::-1]
            dq = dq.transpose()
            dq = dq[::-1]
        else:
            im = np.rot90(im)
            im = im[:, ::-1] # maybe??
            dq = np.rot90(dq)
            dq = dq[:, ::-1]
        origin='upper' # Argh, the FGS DHAS diagnostic plots put 0,0 at the UPPER left
    else:
        origin='lower'

    mean, median, sigma = astropy.stats.sigma_clipped_stats(im, )

    norm = matplotlib.colors.AsinhNorm(vmin = median-sigma, vmax=im.max(), linear_width=im.max()/1e3)

    ax.imshow(im, norm=norm, origin=origin)

    if use_dq:
        bpmask = np.zeros_like(im, float) + np.nan
        bpmask[(dq & 1)==1] = 1
        ax.imshow(bpmask, vmin=0, vmax=1.5, cmap=matplotlib.cm.inferno, origin=origin)

    ax.set_title(os.path.basename(filename))
    #ax.xaxis.set_ticks([])
    #ax.yaxis.set_ticks([])

    if show_metadata:
        ax.text(0.01, 0.99, f'{model.meta.instrument.detector}\nGS index: {model.meta.guidestar.gs_order}\n{model.meta.observation.date_beg[:-4]}',
                color='yellow',
                transform=ax.transAxes,
                verticalalignment='top')
        if orientation=='raw':
            ax.text(0.99, 0.99, f'detector raw orientation',
                color='yellow',
                transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right')

    if return_model:
        return model


def show_all_gs_images(filenames, guidemode='ID'):
    """Show a set of GS ID/Acq images, notionally all from the same visit

    Displays a grid of plots up to 3x3 for the 3 candidates times 3 attempts each

    """

    print(f"Found a total of {len(filenames)} ID images for that observation.")

    ncols= min(3, len(filenames))
    nrows = int(np.ceil(len(filenames)/3))

    print(f'Loading and plotting ID images...')
    fig, axes = plt.subplots(figsize=(16,6*nrows), nrows=nrows, ncols=ncols,
                            gridspec_kw={'wspace': 0.01, 
                                         'left': 0.05,
                                         'right': 0.97,
                                         'top': 0.90,
                                         'bottom': 0.05,
                                         })

    axesf = axes.flatten() if nrows*ncols>1 else [axes,]


    for i, filename in enumerate(filenames):
        if guidemode=='ID':
            display_one_id_image(filename, ax=axesf[i], orientation='raw', count=i)
        else:
            display_one_guider_image(filename, ax=axesf[i], count=i)


        if np.mod(i,3):
            axesf[i].set_yticklabels([])

    for extra_ax in axesf[i+1:]:
        extra_ax.set_axis_off()


def retrieve_and_display_id_images(sci_filename=None, progid=None, obs=None, visit=1, visitid=None, save=True, save_dpi=150):
    """ Top-level routine to retrieve and display FGS ID images for a given visit

    You can specify the visit either by giving a science filename (in which case
    the metadata is read from the header) or directly supplying program ID, observation,
    and optionally visit number.
    """
    # user interface convenience - infer whether the provided string is a visit id automatically
    if sci_filename.startswith('V') and visitid is None:
        visitid = sci_filename
        sci_filename = None

    if visitid is not None:
        progid = int(visitid[1:6])
        obs = int(visitid[6:9])
        visit = int(visitid[9:12])
    filenames = find_guiding_id_file(sci_filename=sci_filename,
                                                      progid=progid, obs=obs, visit=visit)

    show_all_gs_images(filenames)
    visit_id = fits.getheader(filenames[0],ext=0)['VISIT_ID']
    fig = plt.gcf()
    fig.suptitle(f"FGS ID in V{visit_id}", fontsize=16, fontweight='bold')

    if save:
        outname = f'V{visit_id}_ID_images.pdf'
        plt.savefig(outname, dpi=save_dpi, transparent=True)
        print(f"Output saved to {outname}")


def retrieve_and_display_guider_images(visitid=None, progid=None, obs=None, visit=1, guidemode='ACQ1', save=True, save_dpi=150):
    """ Top-level routine to retrieve and display FGS Acq/Track/Fine Guide images for a given visit

    You can specify the visit either by giving a science filename (in which case
    the metadata is read from the header) or directly supplying program ID, observation,
    and optionally visit number.

    Parameters
    ----------
    guidemode : str
        'ACQ1' or 'ACQ2' or 'TRACK' or 'FINEGUIDE'
    """
    if visitid is not None:
        progid = int(visitid[1:6])
        obs = int(visitid[6:9])
        visit = int(visitid[9:12])

    filenames = find_visit_guiding_files(visitid=visitid, guidemode=guidemode,)

    show_all_gs_images(filenames, guidemode=guidemode)
    visit_id = fits.getheader(filenames[0],ext=0)['VISIT_ID']
    fig = plt.gcf()
    fig.suptitle(f"FGS {guidemode} in V{visit_id}", fontsize=16, fontweight='bold')

    if save:
        outname = f'V{visit_id}_{guidemode}_images.pdf'
        plt.savefig(outname, dpi=save_dpi, transparent=True)
        print(f"Output saved to {outname}")


def which_guider_used(visitid, guidemode = 'FINEGUIDE'):
    """ Query MAST for which guider was used in a given visit.

    Parameters
    ----------
    visitid : str
        visit ID, like "V01234004001"
    guidemode : str
        Which kind of guide mode to check. Defaults to FINEGUIDE but
        would need to be TRACK for moving targets

    """
    progid = (visitid[1:6])  # These must be strings
    obs = (visitid[6:9])
    visit = (visitid[9:12])

    keywords = {
    'program': [progid]
    ,'observtn': [obs]
    ,'visit': [visit]
    ,'exp_type': ['FGS_'+guidemode]
    }

    params = {
    'columns': '*',
    'filters': set_params(keywords)
    }

    # Run the web service query. This uses the specialized, lower-level webservice for the
    # guidestar queries: https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar

    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)

    # check the APERNAME which should be either the string FGS1_FULL or FGS2_FULL.
    # All guiding in a visit will use the same guide star, so it's sufficiient to just check the first one 
    if len(t) > 0:
        guider_used = t['apername'][0][0:4]
    else:
        guider_used = None
    return guider_used

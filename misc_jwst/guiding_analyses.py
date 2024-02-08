
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

def mast_retrieve_guiding_files(filenames, out_dir='.'):
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
            print("ERROR: " + outfile + " failed to download.")
        else:
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
    print(delta_times)
    delta_times[delta_times<0] = np.nan
    print(delta_times)

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

def guiding_performance_plot(sci_filename, verbose=True, save=False, yrange=None):
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
    fig, axes = plt.subplots(figsize=(16,12), nrows=3)

    min_time = np.min([ptimes.plot_date.min(), ctimes.plot_date.min()])
    max_time = np.max([ptimes.plot_date.max(), ctimes.plot_date.max()])
    dtime = max_time - min_time

    # Find the subset of centroid data from during that exposure
    ptimes_during_exposure = (t_beg < ptimes ) & (ptimes < t_end)
    ctimes_during_exposure = (t_beg < ctimes[mask] ) & (ctimes[mask] < t_end)


    for i in range(3):
        axes[i].xaxis.axis_date()
        axes[i].set_xlim(min_time -0.01*dtime, max_time+0.01*dtime)

    axes[0].set_title(f"Guiding during {os.path.basename(sci_filename)}", fontweight='bold', fontsize=18)

    axes[0].semilogy(ptimes.plot_date, pointing_table.columns['jitter'], alpha=1, color='C0')
    axes[0].set_ylim(1e-2, 1e3)
    axes[0].set_ylabel("Jitter\n[mas]", fontsize=18)
    axes[0].text(0.01, 0.95, display_gs_fn, fontsize=16, transform=axes[0].transAxes, verticalalignment='top')
    axes[0].text(t_beg.plot_date, 20,
                 f"    mean jitter: {pointing_table.columns['jitter'][ptimes_during_exposure].mean():.2f} mas", color='green')



    axes[0].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
    axes[0].text(t_beg.plot_date, 50, " Exposure", color='green')
    axes[0].text(ptimes.plot_date.min(), 100, " Guiding Start", color='C0')
    axes[0].text(ptimes.plot_date.max(), 100, "Guiding End ", color='C0',
                horizontalalignment='right')


    axes[1].plot(ctimes[mask].plot_date, centroid_table[mask]['guide_star_position_x']-xmean, label='X Centroids', color='C1')
    axes[1].plot(ctimes[mask].plot_date, centroid_table[mask]['guide_star_position_y']-ymean, label='Y Centroids', color='C4')
    axes[1].legend()
    axes[1].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
    axes[1].set_ylabel("GS centroid offsets\n[arcsec]", fontsize=18)
    axes[1].axhline(0, ls=":", color='gray')
    if yrange is not None:
        axes[1].set_ylim(*yrange)

    axes[2].plot(ctimes.plot_date, mask, label='GOOD Centroids', color='C1')

    axes[2].axvspan(t_beg.plot_date, t_end.plot_date, color='green', alpha=0.15)
    axes[2].set_ylabel("Centroid Quality Flag\n", fontsize=18)
    axes[2].set_yticks((0,1))
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].set_yticklabels(['BAD', 'GOOD'])


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

@functools.lru_cache
def find_guiding_id_file(sci_filename=None, progid=None, obs=None, visit=1, verbose=True):
    """ Given a filename of a JWST science file, retrieve the relevant guiding ID data product.
    This uses FITS keywords in the science header to determine the time period and guide mode,
    and then retrieves the file from MAST

    """


    if sci_filename is not None:
        sci_hdul = fits.open(sci_filename)

        progid_str = sci_hdul[0].header['PROGRAM']
        obs_str = sci_hdul[0].header['OBSERVTN']
        visit_str = sci_hdul[0].header['VISIT_ID']
    elif progid is None or obs is None or visit is None:
        raise RuntimeError("You must supply either the sci_filename parameter, or both progid and obs (optionally also visit)")
    else:
        progid_str = f'{progid:05d}'
        obs_str = f'{obs:03d}'
        visit_str = f'{progid:05d}{obs:03d}{visit:03d}'



    # Set up the query
    keywords = {
    'program': [progid_str]
    ,'observtn': [obs_str]
    ,'visit_id': [visit_str]
    ,'exp_type': ['FGS_ID-IMAGE']
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
        print("Found guiding ID image files:")
        for p in products:
            print("   ", p)


    outfiles = mast_retrieve_guiding_files(products)

    return outfiles



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
        ax.text(0.01, 0.99, f'{model.meta.instrument.detector}\nGS index: {model.meta.guidestar.gs_order}',
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

def show_all_gs_id_images(filenames):
    """Show a set of GS ID images, notionally all from the same visit

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
        display_one_id_image(filename, ax=axesf[i], orientation='raw', count=i)
        if np.mod(i,3):
            axesf[i].set_yticklabels([])

    for extra_ax in axesf[i+1:]:
        extra_ax.set_axis_off()


def retrieve_and_display_id_images(sci_filename=None, progid=None, obs=None, visit=1, save=True, save_dpi=150):
    """ Top-level routine to retrieve and display FGS ID images for a given visit

    You can specify the visit either by giving a science filename (in which case
    the metadata is read from the header) or directly supplying program ID, observation,
    and optionally visit number.
    """
    filenames = find_guiding_id_file(sci_filename=sci_filename,
                                                      progid=progid, obs=obs, visit=visit)

    show_all_gs_id_images(filenames)
    visit_id = fits.getheader(filenames[0],ext=0)['VISIT_ID']
    fig = plt.gcf()
    fig.suptitle(f"FGS ID in V{visit_id}", fontsize=16, fontweight='bold')

    if save:
        outname = f'V{visit_id}_ID_images.pdf'
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

import functools
import os

import matplotlib, matplotlib.pyplot as plt
import numpy as np

import webbpsf
import scipy
import astropy
import astropy.io.fits as fits
import astropy.units as u
import jwst.datamodels, stdatamodels



@functools.lru_cache
def get_visit_ta_image(visitid, verbose=True, kind='cal', inst='NIRCam', index=0):
    """Retrieve from MAST the NIRCam target acq image for a given visit.

    This retrieves an image (or images) from MAST and returns it as a HDUList variable
    without writing to disk.
    - If only one TA file is found, that file is returned directly as an HDUList
    - If multiple TA files ar found (e.g. TACQ and TACONFIRM), a list is returned
      containing all of them.
    """

    from astroquery.mast import Mast
    keywords = {
            'visit_id': [visitid[1:]], # note: drop the initial character 'V'
            'exp_type': ['NRC_TACQ', 'MIR_TACQ', 'NRS_WATA', 'NRS_TACONFIRM', 'NIS_TACQ']
           }

    def set_params(parameters):
        return [{"paramName" : p, "values" : v} for p, v in parameters.items()]


    # Restructuring the keywords dictionary to the MAST syntax
    params = {'columns': '*',
          'filters': set_params(keywords)
         }

    if inst.upper() == 'NIRCAM':
        service = 'Mast.Jwst.Filtered.Nircam'
    elif inst.upper() == 'MIRI':
        service = 'Mast.Jwst.Filtered.Miri'
    elif inst.upper() == 'NIRSPEC':
        service = 'Mast.Jwst.Filtered.NIRSpec'
    elif inst.upper() == 'NIRISS':
        service = 'Mast.Jwst.Filtered.NIRISS'

    if verbose:
        print(f'Querying MAST to find target acq files for {visitid}')
    t = Mast.service_request(service, params)
    nfiles = len(t)
    if verbose:
        print(f"Found {nfiles} target acq files for that observation.")

    files_found = []
    filenames = t['filename']
    filenames.sort()

    for filename in filenames:

        # If user manually specifies rate or uncal, retrieve that instead
        if kind == 'rate' or kind == 'uncal':
             filename = filename.replace('_cal.fits', f'_{kind}.fits')

        if verbose:
            print(f"TA filename: {filename}")
        import urllib
        mast_file_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{filename}"
        localpath = f'./{filename}'
        if os.path.exists(localpath):
            # If that file has been downloaded to this local working dir, open from there
            ta_hdul = fits.open(localpath)
        else:
            try:
                # Try to open from MAST (this version only works for public-access files)
                ta_hdul = fits.open(mast_file_url)
            except urllib.error.HTTPError as err:
                if err.code == 401:  # Unauthorized
                    # Use MAST API to allow retrieval of exclusive access data, if relevant
                    import astroquery
                    import tempfile
                    mast_api_token = os.environ.get('MAST_API_TOKEN', None)
                    mast_obs = astroquery.mast.ObservationsClass(mast_api_token)
                    uri = f"mast:JWST/product/{filename}"
                    mast_obs.download_file(uri, local_path=localpath, cache=False)
                    ta_hdul = astropy.io.fits.open(localpath)
                else:
                    raise  # re-raise any errors other than 401 for permissions denied
        files_found.append(ta_hdul)

    if nfiles==1:
        return files_found[0]
    else:
        return files_found



#### Functions for image comparisons
def show_ta_img(visitid, ax=None, return_handles=False, inst='NIRCam', mark_reference_point=True, mark_apername=True):
    """ Retrieve and display a target acq image"""

    hdul = get_visit_ta_image(visitid, inst=inst)

    ta_img = hdul['SCI'].data
    mask = np.isfinite(ta_img)
    rmean, rmedian, rsig = astropy.stats.sigma_clipped_stats(ta_img[mask])
    bglevel = rmedian

    vmax = np.nanmax(ta_img) - bglevel
    cmap = matplotlib.cm.viridis.copy()
    cmap.set_bad('orange')


    norm = matplotlib.colors.AsinhNorm(linear_width = vmax*0.003, vmax=vmax, #vmin=0)
                                       vmin=-1*rsig)


    if ax is None:
        ax = plt.gca()
    ax.imshow(ta_img - bglevel, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(f"{inst} TA on {visitid}\n{hdul[0].header['DATE-OBS']}")
    ax.set_ylabel("[Pixels]")
    ax.text(0.05, 0.95, hdul[0].header['TARGPROP'],
            color='white', transform=ax.transAxes, verticalalignment='top')

    if mark_reference_point:
        import pysiaf
        siaf = pysiaf.Siaf(hdul[0].header['INSTRUME'])
        ap = siaf.apertures[hdul[0].header['APERNAME']]
        xref_subarr = ap.XSciRef - 1   # siaf uses 1-based counting
        yref_subarr = ap.YSciRef - 1   # ditto

        ax.axvline(xref_subarr, color='gray', alpha=0.5)
        ax.axhline(yref_subarr, color='gray', alpha=0.5)
    if mark_apername:
        # mark aperture, and which guider was used
        ax.text(0.95, 0.95, hdul[0].header['APERNAME']+f"\n using {which_guider_used(visitid)}",
            color='white', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')




    if return_handles:
        return hdul, ax, norm, cmap, bglevel




def nrc_ta_comparison(visitid, inst='NIRCam', verbose=True, show_centroids=True):
    """ Retrieve a NIRCam target acq image and compare to a simulation

    Parameters:
    -----------
    visitid : string
        Visit ID. Should be one of the WFSC visits, starting with a NIRCam target acq, or at least
        some other sort of visit that begins with a NIRCam target acquisition.
    """
    from skimage.registration import phase_cross_correlation

    fig, axes = plt.subplots(figsize=(10,5), ncols=3)

    # Get and plot the observed TA image
    hdul, ax, norm, cmap, bglevel = show_ta_img(visitid, ax=axes[0], return_handles=True, inst=inst)
    im_obs = hdul['SCI'].data
    im_obs_err = hdul['ERR'].data
    im_obs_dq = hdul['DQ'].data

    im_obs_clean = im_obs.copy()
    im_obs_clean[im_obs_dq & 1] = np.nan  # Mask out any DO_NOT_USE pixels.
    im_obs_clean = astropy.convolution.interpolate_replace_nans(im_obs, kernel=np.ones((5,5)))

    # Make a matching sim
    nrc = webbpsf.setup_sim_to_match_file(hdul, verbose=False)
    opdname = nrc.pupilopd[0].header['CORR_ID'] + "-NRCA3_FP1-1.fits"
    if verbose:
        print(f"Calculating PSF to match that TA image...")
    psf = nrc.calc_psf(fov_pixels=im_obs.shape[0])

    # Align and Shift:
    im_sim = psf['DET_DIST'].data   # Use the extension including distortion and IPC

    # apply a mask around the border pixels, to apply a prior that the PSF is probably in the center-ish
    # and ignore any unmasked bad/hot pixels near the edges. This makes this alignment step more robust
    nm = 6
    border_mask = np.ones_like(im_obs_clean)
    border_mask[:nm] = 0
    border_mask[-nm:] = 0
    border_mask[:, :nm] = 0
    border_mask[:, -nm:] = 0
    #axes[0].imshow(im_obs_clean*border_mask , norm=norm, cmap=cmap, origin='lower')

    shift, _, _ = phase_cross_correlation(im_obs_clean*border_mask, im_sim, upsample_factor=32)
    if verbose:
        print(f"Shift to register sim to data: {shift} pix")
    im_sim_shifted = scipy.ndimage.shift(im_sim, shift, order=5)

    # figure out the background level and scale factor
    scalefactor = np.nanmax(im_obs) / im_sim.max()
    if verbose:
        print(f"Scale factor to match sim to data: {scalefactor}")
    im_sim_scaled_aligned = im_sim_shifted*scalefactor


    # Optional, plot the measured centroids
    if show_centroids:
        ### OSS CENTROIDS ###
        # First, see if we can retrieve the on-board TA centroid measurment from the OSS engineering DB in MAST
        try:
            import misc_jwst.engdb
            # retrieve the log for this visit, extract the OSS centroids, and convert to same
            # coordinate frame as used here:
            osslog = misc_jwst.engdb.get_ictm_event_log(hdul[0].header['VSTSTART'], hdul[0].header['VISITEND'])
            try:
                oss_cen = misc_jwst.engdb.extract_oss_TA_centroids(osslog, 'V' + hdul[0].header['VISIT_ID'])
                # Convert from full-frame (as used by OSS) to detector subarray coords:
                oss_cen_sci = nrc._detector_geom_info.aperture.det_to_sci(*oss_cen)
                oss_cen_sci_pythonic = np.asarray(oss_cen_sci) - 1  # convert from 1-based pixel indexing to 0-based
                oss_centroid_text = f"OSS centroid: {oss_cen_sci_pythonic[0]:.2f}, {oss_cen_sci_pythonic[1]:.2f}"
                axes[0].scatter(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], color='0.5', marker='x', s=50)
                axes[0].text(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], 'OSS  ', color='0.9', verticalalignment='center', horizontalalignment='right')
                if verbose:
                    print(f"OSS centroid on board:  {oss_cen}  (full det coord frame, 1-based)")
                    print(f"OSS centroid converted: {oss_cen_sci_pythonic}  (sci frame in {nrc._detector_geom_info.aperture.AperName}, 0-based)")
                    full_ap = nrc.siaf[nrc._detector_geom_info.aperture.AperName[0:5] + "_FULL"]
                    oss_cen_full_sci = np.asarray(full_ap.det_to_sci(*oss_cen)) - 1
                    print(f"OSS centroid converted: {oss_cen_full_sci}  (sci frame in {full_ap.AperName}, 0-based)")

            except RuntimeError:
                if verbose:
                    print("Could not parse TA coordinates from log. TA may have failed?")
                oss_cen_sci_pythonic = None

            ### WCS COORDINATES ###
            model = jwst.datamodels.open(hdul)
            targ_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
            targ_coords_pix = model.meta.wcs.world_to_pixel(targ_coords)  # returns x, y
            axes[0].scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
            axes[0].text(targ_coords_pix[0], targ_coords_pix[1]+2, 'WCS', color='magenta', verticalalignment='bottom', horizontalalignment='center')
            axes[0].text(0.95, 0.04, f'Expected from WCS: {targ_coords_pix[0]:.2f}, {targ_coords_pix[1]:.2f}',
                     horizontalalignment='right', verticalalignment='bottom',
                     transform=axes[0].transAxes,
                     color='white')

            if verbose:
                print(f"Star coords from WCS: {targ_coords_pix}")
                if oss_cen_sci_pythonic is not None:
                    print(f"WCS offset =  {np.asarray(targ_coords_pix) - oss_cen_sci_pythonic} pix")

        except ImportError:
            oss_centroid_text = ""

        ### WEBBPSF CENTROIDS ###
        cen = webbpsf.fwcentroid.fwcentroid(im_obs_clean*border_mask)
        axes[0].scatter(cen[1], cen[0], color='red', marker='+', s=50)
        axes[0].text(cen[1], cen[0], '  webbpsf', color='red', verticalalignment='center')
        axes[0].text(0.95, 0.10, oss_centroid_text+f'\n webbpsf centroid: {cen[1]:.2f}, {cen[0]:.2f}',
                     horizontalalignment='right', verticalalignment='bottom',
                     transform=axes[0].transAxes,
                     color='white')


    # Plot the simulated TA image
    axes[1].imshow(im_sim_scaled_aligned, norm=norm, cmap=cmap, origin='lower')
    axes[1].set_title(f"Simulated PSF in F212N\nusing {opdname}")



    # Plot panel
    diffim = im_obs -bglevel - im_sim_scaled_aligned

    dofs = np.isfinite(diffim).sum() - 4  # 4 estimated parameters: X and Y offsets, flux scaling, background level
    reduced_chisq = np.nansum(((diffim / im_obs_err)**2)) / dofs

    axes[2].imshow(diffim, cmap=cmap, norm=norm, origin='lower')
    axes[2].set_title('Difference image\nafter alignment and scaling')
    axes[2].text(0.05, 0.9, f"$\\chi^2_r$ = {reduced_chisq:.3g}" + (
                  "  Alert, not a good fit!" if (reduced_chisq > 1.5) else ""),
                 transform = axes[2].transAxes, color='white' if (reduced_chisq <1.5) else 'yellow')

    for ax in axes:
        fig.colorbar(ax.images[0], ax=ax, orientation='horizontal',
                    label=hdul['SCI'].header['BUNIT'])

    plt.tight_layout()


    outname = f'nrc_ta_comparison_{visitid}.pdf'
    plt.savefig(outname)
    print(f" => {outname}")



########## NIRSPEC ##########


def plot_full_image(filename_or_datamodel, ax=None, vmax = 10, colorbar=True, colorbar_orientation='vertical'):
    """Plot a full-frame MIRI LRS or NRS WATA image, with annotations"""

    if isinstance(filename_or_datamodel, stdatamodels.jwst.datamodels.JwstDataModel):
        model = filename_or_datamodel
    else:
        model = jwst.datamodels.open(filename_or_datamodel)

    norm = matplotlib.colors.Normalize(vmin=0.1, vmax=vmax)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('orange')

    if ax is None:
        fig = plt.figure(figsize=(16,9))
        ax = plt.gca()

    imcopy = model.data.copy()
    imcopy[(model.dq &1)==1]  = np.nan
    im = ax.imshow(model.data, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(model.meta.filename, fontweight='bold')


    # Metadata annotations

    annotation_text = f"{model.meta.target.proposer_name}\n{model.meta.instrument.filter}, {model.meta.exposure.readpatt}:{model.meta.exposure.ngroups}:{model.meta.exposure.nints}\n{model.meta.exposure.effective_exposure_time:.2f} s"

    try:
        wcs = model.meta.wcs
        # I don't know how to deal with the slightly different API of the GWCS class
        # so, this is crude, just cast it to a regular WCS and drop the high order distortion stuff
        # This suffices for our purposes in plotting compass annotations etc.
        # (There is almost certainly a better way to do this...)
        simple_wcs = astropy.wcs.WCS(model.meta.wcs.to_fits()[0])
    except:
        wcs = model.get_fits_wcs()
        if cube_ints:
            wcs = wcs.dropaxis(2)  # drop the nints axis

    if colorbar:
        # Colorbar

        cb = plt.gcf().colorbar(im, pad=0.1, aspect=60, label=model.meta.bunit_data,
                               orientation=colorbar_orientation)
        cbaxfn = cb.ax.set_xscale if colorbar_orientation=='horizontal' else cb.ax.set_yscale
        cbaxfn('asinh')

    if model.meta.exposure.type =='MIR_TACQ':
        labelstr="Target Acquisition Image"
    elif model.meta.exposure.type =='MIR_TACONFIRM':
        labelstr="Target Acquisition Verification Image"
    else:
        labelstr=""

    ax.set_xlabel("Pixels", fontsize='small')
    ax.set_ylabel("Pixels", fontsize='small')
    ax.tick_params(labelsize='small')
    ax.text(0.01, 0.99, annotation_text,
        transform=ax.transAxes, color='white', verticalalignment='top', fontsize=10)

    ax.text(0.5, 0.99, labelstr,
            style='italic', fontsize=10, color='white',
            horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    try:
        import spaceKLIP
        spaceKLIP.plotting.annotate_scale_bar(ax, model.data, simple_wcs, yf=0.07, xf=0.6)
    except:
        pass

    # Leave off for now, this is not working ideally due to some imviz issue
    #spaceKLIP.plotting.annotate_compass(ax, model.data, wcs, yf=0.07, length_fraction=30)



def nrs_ta_position_fit(model, cutout_center_coords, box_size = 40, plot=False,
                        initial_estimate_stddev=3,
                       use_dq = True):

    # Cut out subregion data, err, dq
    cutout = astropy.nddata.Cutout2D(model.data, cutout_center_coords, box_size)
    cutout_dq = astropy.nddata.Cutout2D(model.dq, cutout_center_coords, box_size)
    cutout_err = astropy.nddata.Cutout2D(model.err, cutout_center_coords, box_size)
    print(cutout.xmin_original)
    if use_dq:
        good  = (cutout_dq.data & 1)==0
    else:
        good = np.isfinite(cutout.data) & (cutout_err.data !=0)

    y, x = np.indices(cutout.data.shape)
    x += cutout.xmin_original
    y += cutout.ymin_original

    med_bg = np.nanmedian(cutout.data)

    g_init = astropy.modeling.models.Gaussian2D(amplitude = np.nanmax(cutout.data),
                                                x_stddev=initial_estimate_stddev, y_stddev=initial_estimate_stddev,
                                      x_mean = cutout_center_coords[0], y_mean=cutout_center_coords[1])
    # Do an initial fit with larger bounds to get a first estimate of center
    g_init.bounds['x_stddev'] = [0.1, 3]
    g_init.bounds['y_stddev'] = [0.1, 3]

    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    result0 = fitter(g_init, x[good], y[good], cutout.data[good]-med_bg,
                    weights = 1./cutout_err.data[good])

    # Do a refit that more precisely constrains the FWHM to something reasonable
    result0.bounds['x_stddev'] = [0.5, 1.5]
    result0.bounds['y_stddev'] = [0.5, 1.5]

    result = fitter(result0, x[good], y[good], cutout.data[good]-med_bg,
                    weights = 1./cutout_err.data[good])

    # Get fit parameters and uncertainties
    covariance = fitter.fit_info['param_cov']
    fitted_params = dict(zip(result.param_names, zip(result.parameters, np.diag(covariance)**0.5)))
    for k in fitted_params:
        print(f"{k} :  \t{fitted_params[k][0]:7.3f} +- {fitted_params[k][1]:.3f} ")


    if plot:
        fig, (ax1,ax2,ax3) = plt.subplots(figsize=(16,9), ncols=3)
        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5, ]

        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(cutout.data),
                                           linear_width=np.nanmax(cutout.data)*0.01)
        ax1.imshow(cutout.data-med_bg,
                   norm=norm,
                   extent=extent)
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent)
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent)
        ax1.set_title("Data Cutout")
        ax2.set_title("Model & Centroids")
        ax3.set_title("Residual vs Gauss2D")
        for ax in (ax1,ax2,ax3):
            ax.plot(result.x_mean, result.y_mean, color='white', marker='x', markersize=20)


    return result, covariance



def nrs_ta_centroids_and_offsets(model, box_size = 16, plot=True, saveplot=True, vmax=1e5, outname_extra="", verbose=True):

    target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
    # target coordinates at epoch, as computed from APT
    print("Target RA, Dec at epoch:")
    print(target_coords)
    target_coords_pix = list(model.meta.wcs.world_to_pixel(target_coords))
    print(target_coords_pix)

    result, covariance = nrs_ta_position_fit(model, target_coords_pix, box_size=box_size,
                                                use_dq=False)


    # Retrieve the OSS onboard centroids for comparison
    import misc_jwst.engdb
    osslog = misc_jwst.engdb.get_ictm_event_log(startdate=model.meta.visit.start_time,
                                    enddate=model.meta.guidestar.visit_end_time)

    oss_centroid = misc_jwst.engdb.extract_oss_TA_centroids(osslog, "V"+model.meta.observation.visit_id, )
    # Y, X flipped here because of NIRSpec detector coords layout?? TBC.
    oss_centroid_conv = [oss_centroid[1] - model.meta.subarray.xstart,
                         oss_centroid[0] - model.meta.subarray.ystart]

    #ta_conf_model.find_fits_keyword('SUBSTRT1'), ta_conf_model.find_fits_keyword('SUBSTRT2')
    if verbose:
        print(f"Onboard OSS TA centroid (1-based): {oss_centroid}")
        print(f"Onboard OSS TA centroid (subarray index): {oss_centroid_conv}")

        print(f"    Agrees with Gaussian fit within: {oss_centroid[0]-1-result.x_mean:.4f}\t {oss_centroid[1]-1-result.y_mean:.4f} pix")

    # offset WCS - oSS
    wcs_offset = [target_coords_pix[i] - (oss_centroid[i] - 1) for i in range(2)]
    wcs_offset = np.asarray(target_coords_pix) - np.asarray(oss_centroid_conv)
    if verbose:
        print(f"WCS offset relative to OSS: {wcs_offset}")

    # offset WCS - Gaussian
    wcs_offset_g = [target_coords_pix[0] - result.x_mean, target_coords_pix[1] - result.y_mean, ]
    if verbose:
        print(f"WCS offset relative to Gaussian fit: {wcs_offset_g}")


    if plot:

        # re-create some values we will need for the plot
        # this partially duplicates code from simple_position_fit for modularity
        cutout = astropy.nddata.Cutout2D(model.data, target_coords_pix, box_size)
        med_bg = np.nanmedian(cutout.data)
        y, x = np.indices(cutout.data.shape)
        x += cutout.xmin_original
        y += cutout.ymin_original



        # Now do some plots
        figscale=0.75
        fig = plt.figure(figsize=(16*figscale, 9*figscale))

        gs = matplotlib.gridspec.GridSpec(3, 4, figure=fig,
                      left=0.04, right=0.75, bottom=0.05, top=0.95,
                     hspace=0.4, wspace=0.15)
        ax0 = fig.add_subplot(gs[:, 0:3])
        ax1 = fig.add_subplot(gs[0, 3])
        ax2 = fig.add_subplot(gs[1, 3])
        ax3 = fig.add_subplot(gs[2, 3])

        ax0.add_artist(matplotlib.patches.Rectangle((cutout.xmin_original, cutout.ymin_original), box_size, box_size,
                            edgecolor='yellow', facecolor='none'))

        plot_full_image(model, ax=ax0, colorbar=False, vmax=vmax)
        if model.meta.filename == 'contents':
            # put a better more descriptive label on the plot
            ax0.set_title(f'NIRSpec {model.meta.exposure.type} for V{model.meta.observation.visit_id}')

        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5, ]

        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(cutout.data),
                                           linear_width=np.nanmax(cutout.data)*0.01)
        ax1.imshow(cutout.data-med_bg,
                   norm=norm,
                   extent=extent)
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent)
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent)
        ax1.set_title("Data Cutout")
        ax2.set_title("Gauss2D Model")
        ax3.set_title("Residual vs Gauss2D")

        if 'WATA' in model.meta.exposure.type:
            # it only makes sense to plot the OSS centroids on the WATA image, not the TACONFIRM one

            ax0.plot(oss_centroid_conv[0], oss_centroid_conv[1],
                     marker='x', color='0.5', markersize=20, ls='none',
                     label='OSS centroid, on board')

            ax0.text(oss_centroid_conv[0]-1, oss_centroid_conv[1], 'OSS     ', color='0.7', verticalalignment='center', horizontalalignment='right')

        ax0.plot(result.x_mean, result.y_mean,
             marker='+', color='orange', markersize=30, ls='none',
             label='using Photutils Gaussian2D')

        ax0.plot(target_coords_pix[0], target_coords_pix[1],
             marker='+', color='magenta', markersize=30, ls='none',
             label='using WCS coords')

        ax0.text(target_coords_pix[0], target_coords_pix[1]-2, 'WCS', color='magenta', verticalalignment='top', horizontalalignment='center')
        ax0.text(result.x_mean.value, result.y_mean.value+2, 'Photutils\nGauss2D', color='orange', verticalalignment='bottom', horizontalalignment='center')

        ax0.legend(facecolor='none', frameon=True, labelcolor='white',
                  markerscale=0.5)

        cprint = lambda xy: f"{xy[0]:.3f}, {xy[1]:.3f}"

        yt = 0.88
        line=0.03
        for i, (label, val) in enumerate((('OSS, onboard', [o-1 for o in oss_centroid]),
                                          ('OSS, subarr', oss_centroid_conv),
                           ('Gaussian2D', (result.x_mean.value, result.y_mean.value)),
                           ('WCS', target_coords_pix))):
            if 'WATA' not in model.meta.exposure.type and 'OSS' in label:
                continue

            fig.text(0.75, yt-line*i, f"{label+':':12s}{cprint(val)}", fontfamily='monospace')
        fig.text(0.75, yt+2*line, "TA Target Coords (pixels):", fontsize=14, fontweight='bold')

        if 'WATA' in model.meta.exposure.type:
            fig.text(0.75, yt-line*6, f"{'Offset WCS-OSS:':12s}{cprint(wcs_offset)}", fontfamily='monospace')
        fig.text(0.75, yt-line*7, f"{'Offset WCS-Gaussian:':12s}{cprint(wcs_offset_g)}", fontfamily='monospace')


        if saveplot:
            label = 'wata' if 'WATA' in model.meta.exposure.type else 'taconfirm'
            outname = f'nrs_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}_{label}_wcs_offset{outname_extra}.pdf'
            plt.savefig(outname)
            print(f" => Saved to {outname}")

    return result, covariance, wcs_offset


def nirspec_wata_ta_analysis(visitid, verbose=True):
    """ Top-level function for NIRSpec WATA TA post-analyses

    Retrieves TACQ and TACONFIRM files from MAST, and runs the position fit and plotting code on each.
    """

    ta_files = get_visit_ta_image(visitid, inst='nirspec', verbose=verbose)
    if len(ta_files) != 2:
        print(f'Warning, expected to find 2 files (TACQ+TACONFIRM) but instead found {len(ta_files)}... ')

    tacq_model = jwst.datamodels.open(ta_files[0])
    taconfirm_model = jwst.datamodels.open(ta_files[1])

    nrs_ta_centroids_and_offsets(tacq_model, verbose=verbose)
    nrs_ta_centroids_and_offsets(taconfirm_model, verbose=verbose)



def nirspec_wata_ta_comparison(visitid, verbose=True, show_centroids=True):
    plt.tight_layout()
    outname = f'nrs_ta_comparison_{visitid}.pdf'
    plt.savefig(outname)
    print(f" => {outname}")




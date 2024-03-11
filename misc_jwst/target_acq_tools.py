import functools
import os

import matplotlib, matplotlib.pyplot as plt
import numpy as np

import webbpsf
import scipy
import astropy
import astropy.io.fits as fits



@functools.lru_cache
def get_visit_ta_image(visitid, verbose=True, kind='cal', inst='NIRCam', index=0):
    """Retrieve from MAST the NIRCam target acq image for a given visit.

    This retrieves an image from MAST and returns it as a HDUList variable
    without writing to disk.
    """

    from astroquery.mast import Mast
    keywords = {
            'visit_id': [visitid[1:]], # note: drop the initial character 'V'
            'exp_type': ['NRC_TACQ', 'MIR_TACQ']
           }

    def set_params(parameters):
        return [{"paramName" : p, "values" : v} for p, v in parameters.items()]


    # Restructuring the keywords dictionary to the MAST syntax
    params = {'columns': '*',
          'filters': set_params(keywords)
         }

    if inst.upper() == 'NIRCAM':
        service = 'Mast.Jwst.Filtered.Nircam'
    if inst.upper() == 'MIRI':
        service = 'Mast.Jwst.Filtered.Miri'
    t = Mast.service_request(service, params)
    nfiles = len(t)
    if verbose:
        print(f"Found {nfiles} target acq files for that observation.")
    filename = t[0]['filename']
    print(filename)

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

    return ta_hdul



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
        import misc_jwst.guiding_analyses
        ax.text(0.95, 0.95, hdul[0].header['APERNAME']+f"\n using {misc_jwst.guiding_analyses.which_guider_used(visitid)}",
            color='white', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')




    if return_handles:
        return hdul, ax, norm, cmap, bglevel




def nrc_ta_image_comparison(visitid, inst='NIRCam', verbose=True, show_centroids=True):
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
            import astropy.units as u
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
            import jwst.datamodels
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

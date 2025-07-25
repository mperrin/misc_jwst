import functools, os

import matplotlib, matplotlib.pyplot as plt
import numpy as np

import pysiaf
import webbpsf
import scipy
import astropy
import astropy.io.fits as fits
import astropy.units as u
import jwst.datamodels, stdatamodels

from misc_jwst import utils
import misc_jwst.engdb


# Constants and lookup tables:

# NIRCam TA Dither Offsets.
# Values from /PRDOPSSOC-067/TA_dithersXML/Excel/NIRCam_TA_dithers.xlsx
# Offsets in detector pixels, in the DET frame
#                        dither_num: [delta_detX, delta_detY]
_ta_dither_offsets_pix = {'NIRCAM': {0: [0,0],
                                     1: [2, 4],
                                     2: [2+3, 4-5]},
                          'NIRISS': {0: [0,0],
                                     1: [4, 2],
                                     2: [4-5, 2+3]},
                          }

# sign convention for DET frame relative to SCI frame
_ta_dither_sign = {'NRCALONG': [1, -1],
                   'NRCBLONG': [1,1],  # TODO check this. But NRC TSO TA isn't dithered.
                    'NRCA1': [1,1],    # TODO check this. But not directly relevant since thus far there have been no dithered TAs on NRCA2
                    'NRCA2': [1,1],    # TODO check this. But not directly relevant since thus far there have been no dithered TAs on NRCA2
                    'NRCA3': [1,1],    # TODO check this. But not directly relevant since WFS TA doesn't use dithers
                    'NRCA4': [1,1],    # TODO check this. But not directly relevant since WFS TA doesn't use dithers
                    'NIS': [1,1],      # TODO check this.
                   }


@functools.lru_cache
def get_visit_ta_image(visitid, verbose=True, kind='cal', inst='NIRCam', index=0, localpath=None, save_localpath=False):
    """Retrieve from MAST the NIRCam target acq image for a given visit.

    This retrieves an image (or images) from MAST and returns it as a HDUList variable
    without writing to disk. If you want it to save to disk, set save_localpath=True

    - If only one TA file is found, that file is returned directly as an HDUList
    - If multiple TA files ar found (e.g. TACQ and TACONFIRM), a list is returned
      containing all of them.
    """

    from astroquery.mast import Mast
    keywords = {
            'visit_id': [visitid[1:]], # note: drop the initial character 'V'
            'exp_type': ['NRC_TACQ', 'MIR_TACQ', 'MIR_TACONFIRM', 'NRS_WATA', 'NRS_TACONFIRM', 'NIS_TACQ']
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
        # use distinct varible names for localpath (input) and local_file_cache_path (internal variable, per filename)
        local_file_cache_path = localpath if localpath is not None else f'./{filename}'
        if os.path.exists(local_file_cache_path):
            # If that file has been downloaded to this local working dir, open from there
            if verbose:
                print("Opening file from local path "+local_file_cache_path)
            ta_hdul = fits.open(local_file_cache_path)
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
                    mast_obs.download_file(uri, local_path=local_file_cache_path, cache=False)
                    ta_hdul = astropy.io.fits.open(local_file_cache_path)
                else:
                    raise  # re-raise any errors other than 401 for permissions denied
            if save_localpath:
                local_file_cache_path = localpath if localpath is not None else f'./{filename}'
                ta_hdul.writeto(local_file_cache_path)
                print(f"Saved to {local_file_cache_path}")

        files_found.append(ta_hdul)

    if nfiles==1:
        #if verbose:
        #    print("File date:", ta_hdul[0].header['DATE'])
        return files_found[0]
    else:
        return files_found


#### Functions for image comparisons

def _crop_display_for_miri(ax, hdul):
    """ Set xlim and ylim to crop the display region for a MIRI TA image
    Many MIRI TA images use full array, even though only a subarray is of interest.
    """
    # Note, this would be more elegant to look up from siaf but we just hard-code values here since none of this will ever change.

    apname = hdul[0].header['APERNAME']

    boxsize=64
    if apname =='MIRIM_TAMRS':
        # crop to upper corner
        ax.set_xlim(1023-boxsize, 1023)
        ax.set_ylim(1019-boxsize, 1019)
    elif apname == 'MIRIM_TASLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
        # crop to upper part.
        ax.set_ylim(415-boxsize, 415)
        ax.set_xlim(8,)
    elif apname == 'MIRIM_SLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
        # crop to upper part.
        ax.set_ylim(415-boxsize-80, 415-80)
        ax.set_xlim(8,)
    elif apname =='MIRIM_TALRS':
        ax.set_xlim(382, 382+boxsize )
        ax.set_ylim(258, 258+boxsize)
    elif apname =='MIRIM_SLIT':
        ax.set_xlim(294, 294+boxsize )
        ax.set_ylim(269, 269+boxsize)


def _crop_display_for_nirspec(ax, hdul):
    """ Set xlim and ylim to crop the display region for a MIRI TA image
    Many MIRI WATA images use full array or a larger subarray, even though only a subarray is of interest.
    """
    # Note, this would be more elegant to look up from siaf but we just hard-code values here since none of this will ever change.
    # Plus, NIRSpec subarray parameters turn out not to be in SIAF at all!

    apname = hdul[0].header['APERNAME']
    subarray_name = hdul[0].header['SUBARRAY']

    if apname=='NRS_S1600A1_SLIT':
        if subarray_name == 'SUB32':
            pass # no cropping needed
        elif subarray_name == 'SUB2048':
            # crop displayed region to be consistent with the SUB32 view
            ax.set_xlim(1398-0.5, 1398+32-0.5)
            # no ylim crop needed, this is already only 32 pix tall
        elif subarray_name == 'FULL':
            # crop displayed region to be consistent with the SUB32 view
            ax.set_xlim(1398-0.5, 1398+32-0.5)
            ax.set_ylim(974-0.5, 975+32-0.5)


def show_ta_img(visitid, ax=None, return_handles=False, inst='NIRCam', mark_reference_point=True, mark_apername=True, ta_expnum=None, **kwargs):
    """ Retrieve and display a target acq image"""

    hdul = get_visit_ta_image(visitid, inst=inst, **kwargs)
    inst = inst.upper()

    # If there are multiple TA images, you have to pick which one you want to display
    if len(hdul) == 0:
        raise RuntimeError("No TA images found for that visit")
    title_extra = ''
    if isinstance(hdul, list) and not isinstance(hdul, fits.HDUList):
        if ta_expnum is None:
            raise ValueError(f"You must specify ta_expnum=<n> to select which of {len(hdul)} TA exposures to show (using 1-based indexing)")
        else:
            hdul = hdul[ta_expnum-1]
            if inst.upper() not in ['NIRCAM', 'NIRISS'] :
                # These can have dithered 3x TAs, so print the exp number on each for those cases
                title_extra = f' exp #{ta_expnum}'
            if inst.upper() == 'NIRSPEC':
                if 'WATA' in hdul[0].header['EXP_TYPE']:
                    title_extra = ' (WATA)'
                elif 'MSATA' in hdul[0].header['EXP_TYPE']:
                    title_extra = ' (MSATA)'
            if 'CONF' in hdul[0].header['EXP_TYPE']:
                title_extra = 'CONFIRM'


    ta_img = hdul['SCI'].data
    mask = np.isfinite(ta_img)
    rmean, rmedian, rsig = astropy.stats.sigma_clipped_stats(ta_img[mask])
    bglevel = rmedian

    vmax = np.nanmax(ta_img) - bglevel
    asinh_linear_width = vmax*0.003
    # special case NIRSpec
    if inst.upper() == 'NIRSPEC' and hdul[0].header['SUBARRAY'] == 'FULL':
        # set vmin, vmax just based on the region around the S1600 aperture for WATA
        vmax = np.nanmax(ta_img[974:974+32, 1399:1399+32]) - bglevel
        # set linear width based on bg level inside the S1600 aperture
        asinh_linear_width = np.max([np.nanmedian(ta_img[1407:1407+16, 983:983+16])*2, vmax*0.003])
    cmap = matplotlib.cm.viridis.copy()
    cmap.set_bad('orange')


    print('vmin, vmax, asinh_linear', -1*rsig, vmax, asinh_linear_width)
    norm = matplotlib.colors.AsinhNorm(linear_width = asinh_linear_width, vmax=vmax, #vmin=0)
                                       vmin=-1*rsig)

    model = jwst.datamodels.open(hdul)
    annotate_subarray = f"\n{model.meta.subarray.name} " if inst.upper() == "NIRSPEC" else ""
    annotation_text = f"{model.meta.target.proposer_name}\n{model.meta.instrument.filter}, {model.meta.exposure.readpatt}:{model.meta.exposure.ngroups}:{model.meta.exposure.nints}{annotate_subarray}\n{model.meta.exposure.effective_exposure_time:.2f} s"

    if ax is None:
        ax = plt.gca()
    ax.imshow(ta_img - bglevel, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(f"{inst} TA{title_extra} on {visitid}\n{hdul[0].header['DATE-OBS']} {hdul[0].header['TIME-OBS'][0:8]}")
    ax.set_ylabel("[Pixels]")
    ax.text(0.05, 0.95, annotation_text,
            color='white', transform=ax.transAxes, verticalalignment='top')

    if inst.upper() == 'MIRI':
        _crop_display_for_miri(ax, hdul)
    elif inst.upper() == 'NIRSPEC':
        _crop_display_for_nirspec(ax, hdul)


    if mark_reference_point:
        xref, yref = get_ta_reference_point(inst, hdul, ta_expnum)
        ax.axvline(xref, color='0.75', alpha=0.5, ls='--')
        ax.axhline(yref, color='0.75', alpha=0.5, ls='--')

    if mark_apername:
        # mark aperture, and which guider was used
        import misc_jwst.guiding_analyses
        ax.text(0.95, 0.95, hdul[0].header['APERNAME']+f"\n using {misc_jwst.guiding_analyses.which_guider_used(visitid)}",
            color='white', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')


    if return_handles:
        return hdul, ax, norm, cmap, bglevel


def get_ta_reference_point(inst, hdul, ta_expnum=1):
    """ Determine the TA reference point, i.e. where was the target 'supposed to be' in any given observation
    This is usually the aperture reference location, but may be modified by any TA dithers, or
    by subarray/full array coord system shenanigans for MIR.
    """
    siaf = utils.get_siaf(inst)
    ap = siaf.apertures[hdul[0].header['APERNAME']]

    if inst.upper() in ['NIRCAM', 'NIRISS']:
        xref = ap.XSciRef - 1   # siaf uses 1-based counting
        yref = ap.YSciRef - 1   # ditto

        if ta_expnum>0:
            # if there are multiple ta exposures, take into account the dither moves
            # and the sign needed to go from DET to SCI coordinate frames
            xref -= _ta_dither_offsets_pix[inst][ta_expnum-1][0] * _ta_dither_sign[hdul[0].header['DETECTOR']][0]
            yref -= _ta_dither_offsets_pix[inst][ta_expnum-1][1] * _ta_dither_sign[hdul[0].header['DETECTOR']][1]
    elif inst.upper() == 'MIRI':
        try:
            if hdul[0].header['SUBARRAY'] == 'FULL' and hdul[0].header['APERNAME'] != 'MIRIM_SLIT':
                # For MIRI, sometimes we have subarray apertures for TA but the images are actually full array.
                # In which case use the Det coords
                xref = ap.XDetRef - 1   # siaf uses 1-based counting
                yref = ap.YDetRef - 1   # ditto
            elif hdul[0].header['APERNAME'] == 'MIRIM_TASLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
                # This one's weird. It's pointed using TASLITLESSPRISM but read out usng SLITLESSPRISM
                ap_slitlessprism = siaf.apertures['MIRIM_SLITLESSPRISM']
                xref, yref = ap_slitlessprism.det_to_sci(ap.XDetRef, ap.YDetRef)  # convert coords from TASLITLESSPRISM to SLITLESSPRISM
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting
            elif hdul[0].header['APERNAME'] == 'MIRIM_SLIT':
                # This case is tricky. TACONFIRM image in slit type aperture, which doesn't have Det or Sci coords defined
                # It's made even more complex by filter-dependent MIRI offsets
                print("TODO need to get more authoritative intended target position for MIRIM TA CONFIRM")
                xref, yref = 317, 301
            elif hdul[0].header['APERNAME'].endswith('_UR') or  hdul[0].header['APERNAME'].endswith('_CUR'):
                # Coronagraphic TA, pointed using special TA subarrays but read out using the full coronagraphic subarray
                # Similar to how TASLITLESSPRISM works
                ap_subarray = siaf.apertures['MIRIM_'+hdul[0].header['SUBARRAY']]
                xref, yref = ap_subarray.det_to_sci(ap.XDetRef, ap.YDetRef)  # convert coords from TA subarray to coron subarray
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting


            else:
                xref = ap.XSciRef - 1   # siaf uses 1-based counting
                yref = ap.YSciRef - 1   # ditto


        except TypeError: # LRS slit type doesn't have X/YSciRef
            xref = yref = 0
            print('ERROR DEBUG THIS')

    return xref, yref



########## NIRCAM ##########

def nrc_ta_comparison(visitid, inst='NIRCam', verbose=True, show_centroids=True, **kwargs):
    """ Retrieve a NIRCam target acq image and compare to a simulation

    Parameters:
    -----------
    visitid : string
        Visit ID. Should be one of the WFSC visits, starting with a NIRCam target acq, or at least
        some other sort of visit that begins with a NIRCam target acquisition.

    By default it downloads the file from MAST, or looks in the local directory to see if already downloaded.
    Set localpath=[some path] to search for the file in some other directory or filename.
    """
    from skimage.registration import phase_cross_correlation
    visitid = utils.get_visitid(visitid)

    fig, axes = plt.subplots(figsize=(10,5), ncols=2 if inst.upper() == 'MIRI' else 3)

    # Get and plot the observed TA image
    hdul, ax, norm, cmap, bglevel = show_ta_img(visitid, ax=axes[0], return_handles=True, inst=inst, **kwargs)
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
                oss_centroid_text = "No OSS centroid; TA failed"

            ### WCS COORDINATES ###
            model = jwst.datamodels.open(hdul)
            targ_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
            targ_coords_pix = model.meta.wcs.world_to_pixel(targ_coords)  # returns x, y
            if verbose:
                print(f"Target coords: {targ_coords}")
                print(f"               {targ_coords.to_string('hmsdms', sep=':')}")
            axes[0].scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
            axes[0].text(targ_coords_pix[0], targ_coords_pix[1]+2, 'WCS', color='magenta', verticalalignment='bottom', horizontalalignment='center')
            axes[0].text(0.95, 0.04, f'Expected from WCS: {targ_coords_pix[0]:.2f}, {targ_coords_pix[1]:.2f}',
                     horizontalalignment='right', verticalalignment='bottom',
                     transform=axes[0].transAxes,
                     color='white')

            if verbose:
                print(f"Star coords from WCS: {targ_coords_pix}")
                if oss_cen_sci_pythonic is not None:
                    print(f"WCS offset =  {np.asarray(targ_coords_pix) - oss_cen_sci_pythonic} pix  (WCS - OSS)")

        except ImportError:
            oss_centroid_text = ""

        ### WEBBPSF CENTROIDS ###
        cen = webbpsf.fwcentroid.fwcentroid(im_obs_clean*border_mask)
        axes[0].scatter(cen[1], cen[0], color='red', marker='+', s=50)
        axes[0].text(cen[1], cen[0], '  webbpsf', color='red', verticalalignment='center')

        axes[0].text(0.95, 0.10, oss_centroid_text+f'\n webbpsf measure_centroid: {cen[1]:.2f}, {cen[0]:.2f}',
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


def nrc_ta_analysis(visitid, inst='NIRCam', verbose=True,  plot=True, return_wcs_offsets=False, return_pointing_offsets=False, **kwargs):
    """ Retrieve a NIRCam or NIRISS target acq image, and analyze TA performance.

    See also nrc_ta_comparison

    Parameters:
    -----------
    visitid : string
        Visit ID. Should be one of the WFSC visits, starting with a NIRCam target acq, or at least
        some other sort of visit that begins with a NIRCam target acquisition.
    return_wcs_offsets : bool
        return WCS offets, i.e. the offset between where OSS measured the target and where WCS says the target should be
        Incompatible with return_pointing_offsets
    return_pointing_offsets: bool
        return pointing offsets, i.e. the offset between where OSS measured the target and where it was intended to be
        Incompatible with return_wcs_offsets

    By default it downloads the file from MAST, or looks in the local directory to see if already downloaded.
    Set localpath=[some path] to search for the file in some other directory or filename.
    """
    visitid = utils.get_visitid(visitid)

    inst = inst.upper()
    siaf = utils.get_siaf(inst)
    ta_images = get_visit_ta_image(visitid, inst=inst, **kwargs)
    if ta_images is None:
        raise RuntimeError(f"No TA image found for visit {visitid}")
    elif isinstance(ta_images, astropy.io.fits.HDUList):
        n_ta_images = 1
    elif isinstance(ta_images[0], astropy.io.fits.HDUList):
        n_ta_images = len(ta_images)
    if verbose:
        print(f"Found {n_ta_images} TA images for {visitid} ")

    if plot:
        fig, axes = plt.subplots(figsize=(12,5.5), ncols=2 if inst.upper() == 'MIRI' else 3)

    # Get and plot the observed TA image
    for i_ta_image in range(n_ta_images):

        if plot:
            hdul, ax, norm, cmap, bglevel = show_ta_img(visitid, ax=axes[i_ta_image], return_handles=True, inst=inst,
                                                        ta_expnum = i_ta_image +1, **kwargs)
        else:
            hdul = get_visit_ta_image(visitid, inst=inst)
            if isinstance(hdul, list) and not isinstance(hdul, fits.HDUList):
                          hdul = hdul[i_ta_image]
        ta_aperture = siaf.apertures[hdul[0].header['APERNAME']]
        xref, yref = get_ta_reference_point(inst, hdul, i_ta_image+1)

        if inst.upper() == 'NIRCAM':
            full_ap = siaf[ta_aperture.AperName[0:5] + "_FULL"]
            interp_kernel_size = (5,5)
            show_oss_for_image = n_ta_images-1  # TA computed on the *LAST* of N dithered TA exposures
        elif inst.upper() == 'NIRISS':
            full_ap = siaf["NIS_CEN"]
            interp_kernel_size = (5,5)
            show_oss_for_image = n_ta_images-1  # TA computed on the *LAST* of N dithered TA exposures
        elif inst.upper() == 'MIRI':
            full_ap = siaf["MIRIM_FULL"]
            interp_kernel_size = (11,11)  # bigger, to deal with the border pixels around the image
            show_oss_for_image = 0  # TA computed on the *FIRST* exposure (TA), followed by a TACONFIRM

            if hdul[0].header['APERNAME']=='MIRIM_SLIT':
                # This is a slit type aperture which doesn't have pixel transforms!
                # Therefore use the full aperture for coord transforms
                ta_aperture = full_ap

        im_obs = hdul['SCI'].data
        im_obs_dq = hdul['DQ'].data
        im_obs_clean = im_obs.copy()
        im_obs_clean[im_obs_dq & 1] = np.nan  # Mask out any DO_NOT_USE pixels.
        im_obs_clean = astropy.convolution.interpolate_replace_nans(im_obs, kernel=np.ones(interp_kernel_size))

        aperture_text = f'Intended target pos: {xref:.2f}, {yref:.2f}'

        ### OSS CENTROIDS ###
        # First, see if we can retrieve the on-board TA centroid measurment from the OSS engineering DB in MAST
        try:
            # retrieve the log for this visit, extract the OSS centroids, and convert to same
            # coordinate frame as used here:
            osslog = misc_jwst.engdb.get_ictm_event_log(hdul[0].header['VSTSTART'], hdul[0].header['VISITEND'])
            try:
                oss_cen = misc_jwst.engdb.extract_oss_TA_centroids(osslog, 'V' + hdul[0].header['VISIT_ID'])

                if inst.upper() == "NIRISS":
                    if verbose:
                        print("transposing X & Y, due to NIRISS detector coordinate frame")
                    oss_cen = oss_cen[::-1]

                # Convert from full-frame (as used by OSS) to detector subarray coords:
                oss_cen_sci = ta_aperture.det_to_sci(*oss_cen)
                # For MIRI only, deal with the fact that the image data can be full array even if the aperture is subarray
                if inst.upper() == 'MIRI':
                    if hdul[0].header['SUBARRAY'] == 'FULL' :
                        oss_cen_sci = np.asarray(full_ap.det_to_sci(*oss_cen))
                    elif hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
                        # Special case, pointed using TASLITLESSPRISM but read out using SLITLESSPRISM
                        slitlessprism_ap = siaf['MIRIM_SLITLESSPRISM']
                        oss_cen_sci = np.asarray(slitlessprism_ap.det_to_sci(*oss_cen))
                    elif hdul[0].header['SUBARRAY'].startswith('MASK'):
                        coron_ap = siaf['MIRIM_'+ hdul[0].header['SUBARRAY']]
                        oss_cen_sci = np.asarray(coron_ap.det_to_sci(*oss_cen))


                oss_cen_sci_pythonic = np.asarray(oss_cen_sci) - 1  # convert from 1-based pixel indexing to 0-based
                oss_cen_full_sci = np.asarray(full_ap.det_to_sci(*oss_cen)) - 1

                if i_ta_image == show_oss_for_image:
                    # the OSS centroid is computed onboard relative to the LAST of n TA images, if there's more than 1
                    # So if we are showing the last image, then it makes sense to mark and annotate the OSS centroid location.
                    oss_centroid_text = f"OSS centroid: {oss_cen_sci_pythonic[0]:.2f}, {oss_cen_sci_pythonic[1]:.2f}"

                    if plot:
                        axes[i_ta_image].scatter(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], color='0.5',
                                                 marker='x', s=50)
                        axes[i_ta_image].text(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], 'OSS  ', color='0.9',
                                              verticalalignment='center', horizontalalignment='right')
                else:
                    oss_centroid_text = ""

                if verbose:
                    print(f"OSS centroid on board:  {oss_cen}  (full det coord frame, 1-based)")
                    print(f"OSS centroid converted: {oss_cen_sci_pythonic}  (sci frame in {ta_aperture.AperName}, 0-based)")
                    print(f"OSS centroid converted: {oss_cen_full_sci}  (sci frame in {full_ap.AperName}, 0-based)")

            except RuntimeError:
                if verbose:
                    print("Could not parse TA coordinates from log. TA may have failed?")
                oss_cen_sci_pythonic = (np.nan, np.nan)
                oss_centroid_text = "No OSS centroid; TA failed"

            ### WCS COORDINATES ###
            model = jwst.datamodels.open(hdul)
            targ_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
            targ_coords_pix = model.meta.wcs.world_to_pixel(targ_coords)  # returns x, y
            wcs_text = f'Expected from WCS: {targ_coords_pix[0]:.2f}, {targ_coords_pix[1]:.2f}'
            if verbose:
                print(f"Target coords: {targ_coords}")
                print(f"               {targ_coords.to_string('hmsdms', sep=':')}")
            if plot:
                axes[i_ta_image].scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
                axes[i_ta_image].text(targ_coords_pix[0], targ_coords_pix[1]+2, 'WCS', color='magenta', verticalalignment='bottom', horizontalalignment='center')



        except ImportError:
            oss_centroid_text = ""
            wcs_text = ""

        ### WEBBPSF CENTROIDS ###

        # apply a mask around the border pixels, to apply a prior that the PSF is probably in the center-ish
        # and ignore any unmasked bad/hot pixels near the edges. This makes this alignment step more robust
        nm = 6
        border_mask = np.ones_like(im_obs_clean)
        border_mask[:nm] = 0
        border_mask[-nm:] = 0
        border_mask[:, :nm] = 0
        border_mask[:, -nm:] = 0

        if hdul[0].header['APERNAME']=='MIRIM_TALRS':
            # Special case, it's a full frame image but enforce just centroiding in the TA region of interest
            border_mask[:] = 0
            border_mask[260:320, 390:440] = 1
        elif hdul[0].header['APERNAME']=='MIRIM_SLIT':
            border_mask[:] = 0
            border_mask[270:330, 290:350] = 1
        elif hdul[0].header['APERNAME'].endswith('_UR'):
            # coronagraphy upper right quadrant TA
            imin, imax = (200, 250) if 'LYOT' in hdul[0].header['APERNAME'] else (150, 200)
            border_mask[:] = 0
            border_mask[imin:imax, imin:imax] = 1
        elif hdul[0].header['APERNAME'].endswith('_CUR'):
            # coronagraphy center upper right quadrant TA
            imin, imax = (150, 225) if 'LYOT' in hdul[0].header['APERNAME'] else (100, 150)
            border_mask[:] = 0
            border_mask[imin:imax, imin:imax] = 1

        cen = webbpsf.fwcentroid.fwcentroid(im_obs_clean*border_mask)
        if plot:
            axes[i_ta_image].scatter(cen[1], cen[0], color='red', marker='+', s=50)
            axes[i_ta_image].text(cen[1], cen[0], '  webbpsf', color='red', verticalalignment='center')

        if i_ta_image == show_oss_for_image:
            # For this image, we have an OSS centroid and can compare to that
            deltapos = (oss_cen_sci_pythonic[0] - xref,  oss_cen_sci_pythonic[1] - yref)
            deltapos_type = 'OSS - Intended'
        else:
            # if more than 1 image, for the earlier images, or subsequent TACONFIRMs, show the comparison to webbpsf centroids
            deltapos = (cen[1] - xref,  cen[0] - yref)
            deltapos_type = 'fwcentroid-Intended'

            image_text = f"Pixel coordinates (0-based):         \n{oss_centroid_text}\n webbpsf measure_centroid: {cen[1]:.2f}, {cen[0]:.2f}\n{wcs_text}\n{aperture_text}\n$\\Delta$pos ({deltapos_type}): {deltapos[0]:.2f}, {deltapos[1]:.2f}"

            if plot:
                axes[i_ta_image].text(0.95, 0.04, image_text,
                             horizontalalignment='right', verticalalignment='bottom',
                             transform=axes[i_ta_image].transAxes,
                             color='white')

        if verbose:
            print(f"Star coords from WCS: {targ_coords_pix}")
        if oss_cen_sci_pythonic is not None and (i_ta_image == show_oss_for_image):
            wcs_offset_pix = np.asarray(targ_coords_pix) - oss_cen_sci_pythonic

            # compute delta RA and dec. 
            # from above we already have targ_coords as the RA and dec of the target star, at the time of the exposure
            ta_cen_coords = model.meta.wcs.pixel_to_world(*oss_cen_sci_pythonic)
            dra, ddec = ta_cen_coords.spherical_offsets_to(targ_coords)
            wcs_offset_radec = (dra.to(u.arcsec), ddec.to(u.arcsec))

            if verbose:
                print(f"WCS offset =  {wcs_offset_pix} pix  (WCS - OSS)")
                print('TARG_COORDS: ', targ_coords)
                print('TA_CEN_COORDS: ', ta_cen_coords)
                print("DRA, DDEC: ", dra, ddec)

            if plot:
                axes[show_oss_for_image].text(0.95, 0.75,
                                      f'WCS $\\Delta$RA, $\\Delta$Dec =\n {dra.to_value(u.arcsec):.3f}, {ddec.to_value(u.arcsec):.3f} arcsec',
                                      horizontalalignment='right', verticalalignment='bottom',
                                      transform=axes[show_oss_for_image].transAxes,
                                      color='cyan')

    if plot:
        if n_ta_images == 1:
            axes[1].set_visible(False)
            if inst.upper() != 'MIRI':
                axes[2].set_visible(False)
        for ax in axes[0:n_ta_images]:
            cb = fig.colorbar(ax.images[0], ax=ax, orientation='horizontal',
                        label=hdul['SCI'].header['BUNIT'], fraction=0.05, shrink=0.9, pad=0.07)
            ticks = cb.ax.get_xticks()
            cb.ax.set_xticks([t for t in ticks if t>0.1])

        plt.tight_layout()


        outname = f'{inst.lower()}_ta_analysis_{visitid}.pdf'
        plt.savefig(outname)
        print(f" => {outname}")

    if return_wcs_offsets and return_pointing_offsets:
        raise RuntimeError("You can't set both return_wcs_offsets and return_pointing_offsets at the same time.")
    if return_wcs_offsets:
        return wcs_offset_pix, wcs_offset_radec
    elif return_pointing_offsets:
        # Return pointing offset, and which aperture name that was relative to
        return deltapos, hdul[0].header['APERNAME']


def nrc_ta_get_wcs_offset(visitid, inst='NIRCam', verbose=True, **kwargs):
    """ Get the WCS offset only for a given TA exposure, without looking at the image data

    Method:
     - Get aperture name, from MAST metadata
     - Get OSS centroid, from OSS logs
     - Get attitude matrix, from metadata??
         # Not sure this is directly possible...
         ra_v1  = f.meta.pointing.ra_v1
            dec_v1 = f.meta.pointing.dec_v1
            pa_v3  = f.meta.pointing.pa_v3
            am = pysiaf.utils.rotations.attitude(0, 0, ra_v1, dec_v1, pa_v3)

     - Use attitude matrix and aperture name to

    See also nrc_ta_comparison

    Parameters:
    -----------
    visitid : string
        Visit ID. Should be one of the WFSC visits, starting with a NIRCam target acq, or at least
        some other sort of visit that begins with a NIRCam target acquisition.

    By default it downloads the file from MAST, or looks in the local directory to see if already downloaded.
    Set localpath=[some path] to search for the file in some other directory or filename.
    """

    ta_metadata = misc_jwst.mast.jwst_keywords_query('NIRCAM', visit_id='06005077001', exp_type='NRC_TACQ',
                                       columns='filename, apername')

    if ta_metadata is None:
        raise RuntimeError(f"No TA image found for visit {visitid}")
    elif isinstance(ta_images, astropy.io.fits.HDUList):
        n_ta_images = 1
    elif isinstance(ta_images[0], astropy.io.fits.HDUList):
        n_ta_images = len(ta_images)
    print(f"Found {n_ta_images} TA images for {visitid} ")

    fig, axes = plt.subplots(figsize=(12,5), ncols=3,)

    # Get and plot the observed TA image
    for i_ta_image in range(n_ta_images):
        hdul, ax, norm, cmap, bglevel = show_ta_img(visitid, ax=axes[i_ta_image], return_handles=True, inst=inst,
                                                    ta_expnum = i_ta_image +1, **kwargs)

        im_obs = hdul['SCI'].data
        im_obs_err = hdul['ERR'].data
        im_obs_dq = hdul['DQ'].data

        im_obs_clean = im_obs.copy()
        im_obs_clean[im_obs_dq & 1] = np.nan  # Mask out any DO_NOT_USE pixels.
        im_obs_clean = astropy.convolution.interpolate_replace_nans(im_obs, kernel=np.ones((5,5)))

        siaf = utils.get_siaf(hdul[0].header['INSTRUME'])
        ta_aperture = siaf.apertures[hdul[0].header['APERNAME']]
        xref = ta_aperture.XSciRef - 1   # siaf uses 1-based counting
        yref = ta_aperture.YSciRef - 1   # ditto

        if hdul[0].header['INSTRUME'].upper()=='NIRCAM' and i_ta_image>0:
            # if there are multiple ta exposures, take into account the dither moves
            xref -= _nrc_ta_dither_offsets_pix[i_ta_image][0] * _nrc_ta_dither_sign[hdul[0].header['DETECTOR']][0]
            yref -= _nrc_ta_dither_offsets_pix[i_ta_image][1] * _nrc_ta_dither_sign[hdul[0].header['DETECTOR']][1]

        aperture_text = f'Intended target pos: {xref:.2f}, {yref:.2f}'

        # Optional, plot the measured centroids
        if show_centroids:
            ### OSS CENTROIDS ###
            # First, see if we can retrieve the on-board TA centroid measurment from the OSS engineering DB in MAST
            try:
                # retrieve the log for this visit, extract the OSS centroids, and convert to same
                # coordinate frame as used here:
                osslog = misc_jwst.engdb.get_ictm_event_log(hdul[0].header['VSTSTART'], hdul[0].header['VISITEND'])
                try:
                    oss_cen = misc_jwst.engdb.extract_oss_TA_centroids(osslog, 'V' + hdul[0].header['VISIT_ID'])
                    # Convert from full-frame (as used by OSS) to detector subarray coords:
                    oss_cen_sci = ta_aperture.det_to_sci(*oss_cen)
                    oss_cen_sci_pythonic = np.asarray(oss_cen_sci) - 1  # convert from 1-based pixel indexing to 0-based
                    if i_ta_image == n_ta_images -1:
                        # the OSS centroid is computed onboard relative to the LAST of n TA images, if there's more than 1
                        # So if we are showing the last image, then it makes sense to mark and annotate the OSS centroid location.
                        oss_centroid_text = f"OSS centroid: {oss_cen_sci_pythonic[0]:.2f}, {oss_cen_sci_pythonic[1]:.2f}"

                        axes[i_ta_image].scatter(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], color='0.5',
                                                 marker='x', s=50)
                        axes[i_ta_image].text(oss_cen_sci_pythonic[0], oss_cen_sci_pythonic[1], 'OSS  ', color='0.9',
                                              verticalalignment='center', horizontalalignment='right')
                    else:
                        oss_centroid_text = ""

                    if verbose:
                        print(f"OSS centroid on board:  {oss_cen}  (full det coord frame, 1-based)")
                        print(f"OSS centroid converted: {oss_cen_sci_pythonic}  (sci frame in {ta_aperture.AperName}, 0-based)")
                        full_ap = siaf[ta_aperture.AperName[0:5] + "_FULL"]
                        oss_cen_full_sci = np.asarray(full_ap.det_to_sci(*oss_cen)) - 1
                        print(f"OSS centroid converted: {oss_cen_full_sci}  (sci frame in {full_ap.AperName}, 0-based)")

                except RuntimeError:
                    if verbose:
                        print("Could not parse TA coordinates from log. TA may have failed?")
                    oss_cen_sci_pythonic = None
                    oss_centroid_text = "No OSS centroid; TA failed"

                ### WCS COORDINATES ###
                model = jwst.datamodels.open(hdul)
                targ_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
                targ_coords_pix = model.meta.wcs.world_to_pixel(targ_coords)  # returns x, y
                if verbose:
                    print(f"Target coords: {targ_coords}")
                    print(f"               {targ_coords.to_string('hmsdms', sep=':')}")
                axes[i_ta_image].scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
                axes[i_ta_image].text(targ_coords_pix[0], targ_coords_pix[1]+2, 'WCS', color='magenta', verticalalignment='bottom', horizontalalignment='center')
                wcs_text = f'Expected from WCS: {targ_coords_pix[0]:.2f}, {targ_coords_pix[1]:.2f}'



            except ImportError:
                oss_centroid_text = ""
                wcs_text = ""

            ### WEBBPSF CENTROIDS ###

            # apply a mask around the border pixels, to apply a prior that the PSF is probably in the center-ish
            # and ignore any unmasked bad/hot pixels near the edges. This makes this alignment step more robust
            nm = 6
            border_mask = np.ones_like(im_obs_clean)
            border_mask[:nm] = 0
            border_mask[-nm:] = 0
            border_mask[:, :nm] = 0
            border_mask[:, -nm:] = 0

            cen = webbpsf.fwcentroid.fwcentroid(im_obs_clean*border_mask)
            axes[i_ta_image].scatter(cen[1], cen[0], color='red', marker='+', s=50)
            axes[i_ta_image].text(cen[1], cen[0], '  webbpsf', color='red', verticalalignment='center')

            if i_ta_image == show_oss_for_image:
                # For this image, we have an OSS centroid and can compare to that
                deltapos = (oss_cen_sci_pythonic[0] - xref,  oss_cen_sci_pythonic[1] - yref)
                deltapos_type = 'OSS - Intended'
            else:
                # if more than 1 image, for the earlier images, or subsequent TACONFIRMs, show the comparison to webbpsf centroids
                deltapos = (cen[1] - xref,  cen[0] - yref)
                deltapos_type = 'fwcentroid-Intended'

            image_text = f"Pixel coordinates (0-based):         \n{oss_centroid_text}\npoppy fwcentroid: {cen[1]:.2f}, {cen[0]:.2f}\n{wcs_text}\n{aperture_text}\n$\\Delta$pos ({deltapos_type}): {deltapos[0]:.2f}, {deltapos[1]:.2f}"

            axes[i_ta_image].text(0.95, 0.04, image_text,
                         horizontalalignment='right', verticalalignment='bottom',
                         transform=axes[i_ta_image].transAxes,
                         color='white')

    if verbose:
        print(f"Star coords from WCS: {targ_coords_pix}")
        if oss_cen_sci_pythonic is not None:
            print(f"WCS offset =  {np.asarray(targ_coords_pix) - oss_cen_sci_pythonic} pix  (WCS - OSS)")

            # TODO compute delta RA and dec. Use
            # from above we already have targ_coords as the RA and dec of the target star, at the time of the exposure
            ta_cen_coords = model.meta.wcs.pixel_to_world(*oss_cen_sci_pythonic)
            print('TARG_COORDS: ', targ_coords)
            print('TA_CEN_COORDS: ', ta_cen_coords)

            dra, ddec = ta_cen_coords.spherical_offsets_to(targ_coords)
            print("DRA, DDEC: ", dra, ddec)

            axes[i_ta_image].text(0.95, 0.80,
                                  f'WCS $\\Delta$RA, $\\Delta$Dec = {dra.to_value(u.arcsec):.3f}, {ddec.to_value(u.arcsec):.3f} arcsec',
                                  horizontalalignment='right', verticalalignment='bottom',
                                  transform=axes[i_ta_image].transAxes,
                                  color='cyan')

    if n_ta_images == 1:
        axes[1].set_visible(False)
        axes[2].set_visible(False)
    for ax in axes[0:n_ta_images]:
        cb = fig.colorbar(ax.images[0], ax=ax, orientation='horizontal',
                    label=hdul['SCI'].header['BUNIT'], fraction=0.05, shrink=0.9, pad=0.07)
        ticks = cb.ax.get_xticks()
        cb.ax.set_xticks([t for t in ticks if t>0.1])

    plt.tight_layout()


    outname = f'nrc_ta_analysis_{visitid}.pdf'
    plt.savefig(outname)
    print(f" => {outname}")


########## NIRISS AND MIRI ##########
# (these just piggy-backs on the NIRCam code above)

def nis_ta_analysis(visitid,  verbose=True, **kwargs):
    """ Retrieve a NIRISS target acq image, and analyze TA performance.
    See nrc_ta_analysis for further docs """

    return nrc_ta_analysis(visitid, inst='NIRISS', verbose=verbose, **kwargs)

def miri_ta_analysis(visitid,  verbose=True, **kwargs):
    """ Retrieve a MIRI target acq image, and analyze TA performance.
    See nrc_ta_analysis for further docs """

    return nrc_ta_analysis(visitid, inst='MIRI', verbose=verbose, **kwargs)




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
    target_coords_pix = list(model.meta.wcs.world_to_pixel(target_coords))
    if verbose:
        print("Target RA, Dec at epoch:")
        print(target_coords)
        print(target_coords_pix)

    result, covariance = nrs_ta_position_fit(model, target_coords_pix, box_size=box_size,
                                                use_dq=False)


    # Retrieve the OSS onboard centroids for comparison
    osslog = misc_jwst.engdb.get_ictm_event_log(startdate=model.meta.visit.start_time,
                                    enddate=model.meta.guidestar.visit_end_time)

    oss_centroid = misc_jwst.engdb.extract_oss_TA_centroids(osslog, "V"+model.meta.observation.visit_id, )
    # Y, X flipped here because of NIRSpec detector coords layout?? TBC.
    oss_centroid_conv = [oss_centroid[1] - model.meta.subarray.xstart,
                         oss_centroid[0] - model.meta.subarray.ystart]

    #ta_conf_model.find_fits_keyword('SUBSTRT1'), ta_conf_model.find_fits_keyword('SUBSTRT2')
    # offset WCS - oSS
    wcs_offset = [target_coords_pix[i] - (oss_centroid[i] - 1) for i in range(2)]
    wcs_offset = np.asarray(target_coords_pix) - np.asarray(oss_centroid_conv)

    # offset WCS - Gaussian
    wcs_offset_g = [target_coords_pix[0] - result.x_mean, target_coords_pix[1] - result.y_mean, ]
    if verbose:
        print(f"Onboard OSS TA centroid (1-based): {oss_centroid}")
        print(f"Onboard OSS TA centroid (subarray index): {oss_centroid_conv}")

        print(f"    Agrees with Gaussian fit within: {oss_centroid[0]-1-result.x_mean:.4f}\t {oss_centroid[1]-1-result.y_mean:.4f} pix")

        print(f"WCS offset relative to OSS: {wcs_offset}")
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

        # Check subarray name, if necessary working around API changes in datamodels
        try:
            subarray_name = model.meta.exposure.subarray
        except AttributeError:
            subarray_name = model.meta.subarray.name


        if subarray_name =='SUB2048': # NIRSpec
            # crop displayed region to be conssitent with the SUB32 view
            ax0.set_xlim(1398-0.5, 1398+32-0.5)
            # Y axis is already cropped to 32 pixels in this case.
            # TODO also hadnle the FULL TA case ?
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


def nirspec_wata_ta_comparison(visitid, verbose=True):
    """ Top-level function for NIRSpec WATA TA post-analyses and comparison to photutils

    Retrieves TACQ and TACONFIRM files from MAST, and runs the position fit and plotting code on each.
    """

    ta_files = get_visit_ta_image(visitid, inst='nirspec', verbose=verbose)
    if len(ta_files) != 2:
        print(f'Warning, expected to find 2 files (TACQ+TACONFIRM) but instead found {len(ta_files)}... ')

    tacq_model = jwst.datamodels.open(ta_files[0])
    taconfirm_model = jwst.datamodels.open(ta_files[1])

    nrs_ta_centroids_and_offsets(tacq_model, verbose=verbose)
    nrs_ta_centroids_and_offsets(taconfirm_model, verbose=verbose)



def nirspec_wata_ta_analysis(visitid, verbose=True, show_centroids=True):
    """ Top-level function for NIRSpec WATA TA post analysis
    """
    visitid = utils.get_visitid(visitid)

    ta_files = get_visit_ta_image(visitid, inst='nirspec', verbose=verbose)
    # NIRSpec WATA has a TA image and also a TACONFIRM image.
    n_ta_images = len(ta_files)
    if n_ta_images != 2:
        print(f'Warning, expected to find 2 files (TACQ+TACONFIRM) but instead found {n_ta_images}... ')

    fig, axes = plt.subplots(figsize=(12,7), ncols=2)

    box_size=16

    for i_ta_image in range(n_ta_images):
        ax = axes[i_ta_image]

        hdul, ax, norm, cmap, bglevel = show_ta_img(visitid, ax=ax, return_handles=True, ta_expnum=i_ta_image+1, inst='NIRSpec',
                                                   mark_reference_point=False, verbose=True, save_localpath=True)

        model = jwst.datamodels.open(ta_files[i_ta_image])

        target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
        # target coordinates at epoch, as computed from APT
        target_coords_pix = list(model.meta.wcs.world_to_pixel(target_coords))
        if verbose:
            print("Target RA, Dec at epoch:", target_coords)
            print(target_coords_pix)

        # Retrieve the OSS onboard centroids for comparison
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

        # offset WCS - oSS
        wcs_offset = [target_coords_pix[i] - (oss_centroid[i] - 1) for i in range(2)]
        wcs_offset = np.asarray(target_coords_pix) - np.asarray(oss_centroid_conv)
        if verbose:
            print(f"WCS offset relative to OSS: {wcs_offset}")


       # Check subarray name, if necessary working around API changes in datamodels
        try:
            subarray_name = model.meta.exposure.subarray
        except AttributeError:
            subarray_name = model.meta.subarray.name

        # Cutout subregion just around the S1600A1 aperture
        if subarray_name =='SUB2048':
            # crop displayed region to be conssitent with the SUB32 view
            ax.set_xlim(1398-0.5, 1398+32-0.5)
            # Y axis is already cropped to 32 pixels in this case.
            s1600_corner = (1407, 9)
        elif subarray_name== 'SUB32':
            # already cropped, with FITS SUBSTRT = (1399, 975) [1-based indices]
            s1600_corner = (9, 9)
        elif subarray_name== 'FULL':
            s1600_corner = (1407, 974+9)
        cutout_boxsize = 19   # A bit bigger than the S1600 aperture
        s1600_center = [c + (cutout_boxsize-1)/2 for c in s1600_corner]
        cutout = astropy.nddata.Cutout2D(model.data, s1600_center, cutout_boxsize)

        # Now do some plots
        # Mark location of the S1600 square aperture
        s1600_boxsize = 16
        ax.add_artist(matplotlib.patches.Rectangle(s1600_corner, box_size, box_size,
                            edgecolor='yellow', facecolor='none', linestyle='--'))


        if model.meta.filename == 'contents':
            # put a better more descriptive label on the plot
            ax.set_title(f'NIRSpec {model.meta.exposure.type} for V{model.meta.observation.visit_id}')
        if 'WATA' in model.meta.exposure.type:
            # it only makes sense to plot the OSS centroids on the WATA image, not the TACONFIRM one

            oss_centroid_text = f"OSS centroid: {oss_centroid_conv[0]:.2f}, {oss_centroid_conv[1]:.2f}"

            ax.plot(oss_centroid_conv[0], oss_centroid_conv[1],
                     marker='x', color='0.5', markersize=20, ls='none',
                     label='OSS centroid, on board')

            ax.text(oss_centroid_conv[0]-1, oss_centroid_conv[1], 'OSS     ', color='0.7', verticalalignment='center', horizontalalignment='right')
        else:
            oss_centroid_text = ""

        ax.plot(target_coords_pix[0], target_coords_pix[1],
             marker='+', color='magenta', markersize=30, ls='none',
             label='using WCS coords')

        ax.text(target_coords_pix[0], target_coords_pix[1]-2, 'WCS', color='magenta', verticalalignment='top', horizontalalignment='center')


        cen = deltapos = [-1, -1]
        wcs_text = "[TBD]"
        aperture_text = "[TBD]"
        deltapos_type = '[TBD]'


        image_text = f"Pixel coordinates (0-based):         \n{oss_centroid_text}\n poppy fwcentroid: {cen[1]:.2f}, {cen[0]:.2f}\n{wcs_text}\n{aperture_text}\n$\\Delta$pos ({deltapos_type}): {deltapos[0]:.2f}, {deltapos[1]:.2f}"

        axes[i_ta_image].text(0.95, 0.04, image_text,
                     horizontalalignment='right', verticalalignment='bottom',
                     transform=axes[i_ta_image].transAxes,
                             color='white')

    for ax in axes[0:n_ta_images]:
        cb = fig.colorbar(ax.images[0], ax=ax, orientation='horizontal',
                    label=hdul['SCI'].header['BUNIT'], fraction=0.05, shrink=0.9, pad=0.07)
        ticks = cb.ax.get_xticks()
        cb.ax.set_xticks([t for t in ticks if t>0.1])

    plt.tight_layout()


    outname = f'nrs_ta_analysis_{visitid}.pdf'
    plt.savefig(outname)
    print(f" => {outname}")




###### Overall ############

def auto_ta_results_analysis(visitid, verbose=True):
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format

    inst = misc_jwst.mast.visit_which_instrument(visitid)
    if verbose:
        print(f"Visit {visitid} used {inst} as prime instrument")
    if inst == 'NIRCAM':
        return nrc_ta_analysis(visitid, verbose=verbose)
    elif inst == 'NIRISS':
        return nis_ta_analysis(visitid, verbose=verbose)
    elif inst == 'NIRSPEC':
        raise NotImplementedError("Need code to detect NRS TA type")
    elif inst == 'MIRI':
        raise NotImplementedError("Need code from miri_lrs_fm")



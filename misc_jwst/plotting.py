import numpy as np
import ipyaladin
from astropy.coordinates import SkyCoord
import jwst.datamodels
import pysiaf




def compute_stcs_footprint(apertureSiaf, skyRa, skyDec):
    """
    # Take the sky coordinates of the aperture vertices and convert to an STC-S string
    By Brian McLean, adapted from selectSIAF.py
    See https://github.com/spacetelescope/mast_notebooks/blob/main/notebooks/multi_mission/display_footprints/selectSIAF.py
        """
    if (apertureSiaf.AperShape == 'QUAD' or apertureSiaf.AperShape == 'RECT'):
        apertureSregion = 'POLYGON ICRS {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} '.format(
            skyRa[0], skyDec[0], skyRa[1], skyDec[1], skyRa[2], skyDec[2], skyRa[3], skyDec[3])
    elif apertureSiaf.AperShape == 'CIRC':
        radius = apertureSiaf.maj/3600.0
        apertureSregion = 'CIRCLE ICRS {} {} {} '.format(skyRa, skyDec, radius)
    else:
        print('Unsupported shape {}').format(apertureSiaf.AperShape)

    return apertureSregion



def get_aperture_vertices(apertureSiaf):
    """ Find the selected aperture vertices
    also adapted from https://github.com/spacetelescope/mast_notebooks/blob/main/notebooks/multi_mission/display_footprints/selectSIAF.py
    """
    if (apertureSiaf.observatory == 'Roman' and apertureSiaf.AperShape == 'QUAD'):
        xVertices = np.array([apertureSiaf.XIdlVert1, apertureSiaf.XIdlVert2,
                             apertureSiaf.XIdlVert3, apertureSiaf.XIdlVert4])
        yVertices = np.array([apertureSiaf.YIdlVert1, apertureSiaf.YIdlVert2,
                             apertureSiaf.YIdlVert3, apertureSiaf.YIdlVert4])

    if (apertureSiaf.observatory == 'JWST' and apertureSiaf.AperShape == 'QUAD'):
        xVertices = np.array([apertureSiaf.XIdlVert1, apertureSiaf.XIdlVert2,
                             apertureSiaf.XIdlVert3, apertureSiaf.XIdlVert4])
        yVertices = np.array([apertureSiaf.YIdlVert1, apertureSiaf.YIdlVert2,
                             apertureSiaf.YIdlVert3, apertureSiaf.YIdlVert4])

    if (apertureSiaf.observatory == 'HST' and (apertureSiaf.AperShape == 'QUAD' or apertureSiaf.AperShape == 'RECT')):
        xVertices = np.array(
            [apertureSiaf.v1x, apertureSiaf.v2x, apertureSiaf.v3x, apertureSiaf.v4x])
        yVertices = np.array(
            [apertureSiaf.v1y, apertureSiaf.v2y, apertureSiaf.v3y, apertureSiaf.v4y])
    if (apertureSiaf.observatory == 'HST' and apertureSiaf.AperShape == 'CIRC'):
        xVertices = apertureSiaf.V2Ref
        yVertices = apertureSiaf.V3Ref
    if (apertureSiaf.observatory == 'HST' and apertureSiaf.AperShape == 'PICK'):
        print('Unsupported shape ', apertureSiaf.AperShape)
        xVertices = None
        yVertices = None

    return xVertices, yVertices

def apertures_to_stcs_region(apertureList, attmat):
    """ Iterate over list of apertures to call compute_stsc_footprint
    also adapted from https://github.com/spacetelescope/mast_notebooks/blob/main/notebooks/multi_mission/display_footprints/selectSIAF.py
    """
    # Loop through aperture list  (only works for QUAD, RECT, CIRC aperture shapes)
    # Transform to sky coordinates, build footprints for passing to Aladin
    combinedSregion = ''
    for apertureSiaf in apertureList:
        apertureSiaf.set_attitude_matrix(attmat)
        xVertices, yVertices = get_aperture_vertices(apertureSiaf)
        # Skip PICK which do not have vertices (HST/FGS is only instrument affected)
        if (xVertices is not None and yVertices is not None):
            skyRa, skyDec = apertureSiaf.idl_to_sky(xVertices, yVertices)
            apertureSregion = compute_stcs_footprint(apertureSiaf, skyRa, skyDec)
            combinedSregion += apertureSregion

    return combinedSregion



def aladin_setup_plot(datafile, survey="P/2MASS/color", height=800, fov=0.5):
    """Initialize an aladin instance centered on some JWST image

    Note - it seems you MUST have the aladin output instance as the last item in a Jupyter cell to have
    it initialize and display. Use it like this:

    In[]:
       aladin = aladin_setup_plot(datafile)
       aladin

    Then you can use the aladin instance handle in **subsequent** cells to add overlays.
    See https://github.com/cds-astro/ipyaladin/issues/128

    Parameters
    ----------
    survey : str
        Name of an available all-sky HIPS survey in CDS. See http://aladin.cds.unistra.fr/hips/list
        Some good choices are 'P/2MASS/color', 'P/DSS2/color', 'P/allWISE/color'
    """

    with jwst.datamodels.open(datafile) as model:
        # Get target location
        if model.meta.exposure.type.startswith('FGS'):
            target = SkyCoord(model.meta.guidestar.gs_ra, model.meta.guidestar.gs_dec, frame='icrs', unit='deg')
        else:
            target = SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit='deg')

    # Create an aladin instance centered there.
    aladin = ipyaladin.Aladin(height=height, fov=fov, survey=survey, target=target.to_string())
    return aladin

def aladin_annotate_data_and_apertures(aladin, datafile):
    """ Add a data file and siaf apertures

    Use on an aladin instance you already created. See aladin_setup_plot()
    """
    with jwst.datamodels.open(datafile) as model:
        # Get attitude matrix for the observatory orientation during that exposure
        ra_v1  = model.meta.pointing.ra_v1
        dec_v1 = model.meta.pointing.dec_v1
        pa_v3  = model.meta.pointing.pa_v3
        if ra_v1 is not None:
            # The usual case with good pointing metadata. We can use this straightforwardly to get the attitude matrix
            attmat = pysiaf.utils.rotations.attitude(0, 0, ra_v1, dec_v1, pa_v3)

            # Plot datafile
            aladin.add_fits(datafile)
        else:
            # handle the case of guider ID images, for which we don't have those keywords
            # and there are missing values in the header metadata.

            ra_gs  = model.meta.guidestar.gs_ra
            dec_gs = model.meta.guidestar.gs_dec
            pa_gs  = model.meta.guidestar.gs_v3_pa_science

            # use reference star information from planned guide star table to infer the attitude matrix.
            # This works around the invalid or not properly initialized WCS present in the guide star ID files
            for i in range(len(model.planned_star_table)):
                if model.planned_star_table[i]['guide_star_order'] == model.meta.guidestar.gs_order:
                    row = model.planned_star_table[i]

            aper = pysiaf.Siaf('FGS').apertures[model.meta.aperture.name+"_OSS"]  # Yes, use the OSS version of the aperture here
            ref_v2, ref_v3 = aper.idl_to_tel(row['id_x'], row['id_y'])
            attmat = pysiaf.utils.rotations.attitude(ref_v2, ref_v3, row['ra'], row['dec'], pa_gs)

            # Special case, continued. The WCS in the guider ID images is not valid, so let's fix it. 
            print(f"Caution, FGS images currently have incomplete WCS. Attempting to fix position angle to {pa_gs}...")
            import astropy.wcs, astropy.io.fits as fits
            datafile_hdul = fits.open(datafile)
            # Create a WCS object
            wcs = astropy.wcs.WCS(datafile_hdul['SCI'])
            angle_rad = np.deg2rad(-pa_gs)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            rotated_pc = np.dot(rotation_matrix, wcs.celestial.wcs.pc)  # need the .celestial to work around the awkwardness of the len 1 NAXIS3
            #print('PC', wcs.celestial.wcs.pc)
            #print('rotated', rotated_pc)
            datafile_hdul['SCI'].header['PC1_1'] = rotated_pc[0,0]
            datafile_hdul['SCI'].header['PC1_2'] = rotated_pc[0,1]
            datafile_hdul['SCI'].header['PC2_1'] = rotated_pc[1,0]
            datafile_hdul['SCI'].header['PC2_2'] = rotated_pc[1,1]

            aper_sci = pysiaf.Siaf('FGS').apertures[model.meta.aperture.name]  # Do NOT use the OSS version of the aperture here
            aper_sci.set_attitude_matrix(attmat)
            xpix, ypix = aper_sci.sky_to_sci(ra_gs, dec_gs)
            datafile_hdul['SCI'].header['CRPIX1'] = xpix
            datafile_hdul['SCI'].header['CRPIX2'] = ypix
            datafile_hdul['SCI'].header['CRVAL1'] = ra_gs
            datafile_hdul['SCI'].header['CRVAL2'] = dec_gs

            print(xpix, ypix)

            # Plot the version with the fixed WCS
            aladin.add_fits(datafile_hdul)

            # Reset the target marker in aladin to show the guide star
            aladin.target = SkyCoord(model.meta.guidestar.gs_ra, model.meta.guidestar.gs_dec, frame='icrs', unit='deg')

    # Get apertures
    aps_imaging, aps_spectra, aps_coron = pysiaf.siaf.get_main_apertures()

    stcs_region_imaging = apertures_to_stcs_region(aps_imaging, attmat)
    stcs_region_spectra = apertures_to_stcs_region(aps_spectra, attmat)
    stcs_region_coron = apertures_to_stcs_region(aps_coron, attmat)

    # Plot apertures
    aladin.add_overlay_from_stcs(stcs_region_imaging, color="cyan")
    aladin.add_overlay_from_stcs(stcs_region_spectra, color="magenta")
    aladin.add_overlay_from_stcs(stcs_region_coron, color="lime")

    # Aladin will only show the footprint when you mouse over the display or make another change to it.
    aladin.fov = 0.5


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



def aladin_setup_plot(datafile, survey="P/DSS2/color"):
    """Initialize an aladin instance centered on some JWST image

    Note - it seems you MUST have the aladin output instance as the last item in a Jupyter cell to have
    it initialize and display. Use it like this:

    In[]:
       aladin = aladin_setup_plot(datafile)
       aladin

    Then you can use the aladin instance handle in **subsequent** cells to add overlays.
    See https://github.com/cds-astro/ipyaladin/issues/128
    """

    with jwst.datamodels.open(datafile) as model:
        # Get target location
        target = SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit='deg')

    # Create an aladin instance centered there.
    aladin = ipyaladin.Aladin(height=800, fov=0.5, survey=survey, target=target.to_string())
    return aladin

def aladin_annotate_data_and_apertures(aladin, datafile):
    """ Add a data file and siaf apertures

    Use on an aladin instance you already created. See aladin_setup_plot()
    """
    # Plot datafile
    aladin.add_fits(datafile)

    with jwst.datamodels.open(datafile) as model:
        # Get attitude matrix for the observatory orientation during that exposure
        ra_v1  = model.meta.pointing.ra_v1
        dec_v1 = model.meta.pointing.dec_v1
        pa_v3  = model.meta.pointing.pa_v3
        attmat = pysiaf.utils.rotations.attitude(0, 0, ra_v1, dec_v1, pa_v3)

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


import functools
import astropy, astropy.units as u
import jwst.datamodels
from astroquery.jplhorizons import Horizons
import astropy, astropy.coordinates, astropy.io.fits as fits


def get_visitid(visitstr):
    """ Common util function to handle several various kinds of visit specification"""
    if visitstr.startswith("V"):
        # Full visit ID like V04503031001
        return visitstr
    elif ':' in visitstr:
        # This is PPS format visit ID, like 4503:31:1
        parts = [int(p) for p in visitstr.split(':')]
        if len(parts) == 2:
            # if given only like 4503:31, assume the visit number is 1
            parts.append(1)
        return f"V{parts[0]:05d}{parts[1]:03d}{parts[2]:03d}"
    elif len(visitstr) == 11:
        # Full visit ID but without the leading V, like 04503031001
        return 'V'+visitstr



def colormap_viridis_white_background():
    """Return a colormap like viridis, but with solid white at zero for the background
    Based on code at https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
    """
    from matplotlib.colors import LinearSegmentedColormap

    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    return white_viridis

@functools.lru_cache
def get_siaf(inst):
    """ simple wrapper for siaf load, with caching for speed
    Because it takes like 0.2 seconds per instance to load this.
    """
    import pysiaf
    return pysiaf.Siaf(inst)



def get_target_coords(model):
    """Get target coordinates at epoch of observation from metadata
    returns as astropy coordinates"""
    return astropy.coordinates.SkyCoord(model.meta.target.ra,
                                        model.meta.target.dec,
                                        frame='icrs', unit=u.deg)

def get_guidestar_coords(model):
    """Get guidestar coordinates at epoch of observation from metadata
    returns as astropy coordinates"""
    return astropy.coordinates.SkyCoord(model.meta.guidestar.gs_ra,
                                        model.meta.guidestar.gs_dec,
                                        frame='icrs', unit=u.deg)


def get_obsid_for_filenames(model):
    return f'jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}exp{model.meta.observation.exposure_number}'


def get_jwst_sun_pitch(skycoords, time):
    """Measure sun pitch angle for a given target observed from JWST

    Parameters
    ----------
    skycoords : astropy.coordinates.SkyCoord
        Target location on the sky
    time : astropy.time.Time
        Date and time of the observation

    Returns sun pitch angle in the convention used by JWST flight ops,
    which ranges from pitch  +5º = 85º from the sun
                   to pitch -45º = 135º from the sun.
    This sign convention is not intuitive but we use it here for consistency.


    Example usage:
        fn = 'jw01063001001_02201_00001_nis_rate.fits'

        hdul = fits.open(fn)
        obstime = astropy.time.Time(hdul[0].header['DATE-BEG'])

        v1axis = astropy.coordinates.SkyCoord(hdul[1].header['RA_V1'], hdul[1].header['DEC_V1'],unit='deg' )
        # You can also use TARG_RA and TARG_DEC in the primary header for this,
        # but using the V1 axis pointing is more consistent with the onboard calculation by about 0.1 deg or so.

        get_jwst_sun_pitch(v1axis, obstime)
    """

    # Create Horizons object:
    # id='10' is the Sun
    # location='-170' is JWST
    obj = Horizons(id='10',  # Sun
                   location='500@-170',  # JWST relative to solar system barycenter
                   epochs=time.jd)

    eph = obj.ephemerides()
    sun_seen_from_jwst = astropy.coordinates.SkyCoord(eph['RA'][0], eph['DEC'][0], unit='deg')
    return 90 - skycoords.separation(sun_seen_from_jwst).deg


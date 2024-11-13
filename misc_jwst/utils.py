import functools

def get_visitid(visitstr):
    """ Common util function to handle several various kinds of visit specification"""
    if visitstr.startswith("V"):
        return visitstr
    elif ':' in visitstr:
        # This is PPS format visit ID, like 4503:31:1
        parts = [int(p) for p in visitstr.split(':')]
        return f"V{parts[0]:05d}{parts[1]:03d}{parts[2]:03d}"


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

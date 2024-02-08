
import os

import numpy as np

import astropy, astropy.table
from astroquery.mast import Mast
import requests

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


__all__ = ['jwst_keywords_query']

def set_params(parameters):
    """
    Utility function for making dicts used in MAST queries.

    """
    return [{'paramName': p, 'values': v} for p, v in parameters.items()]


def jwst_keywords_query(instrument, columns=None, all_columns=False, verbose=False,  **kwargs):

    svc_table = {'MIRI':'Mast.Jwst.Filtered.Miri',
                 'NIRCAM': 'Mast.Jwst.Filtered.NIRCam',
                 'NIRSPEC': 'Mast.Jwst.Filtered.NIRSpec',
                 'NIRISS': 'Mast.Jwst.Filtered.NIRISS',
                }

    service = svc_table[instrument.upper()]

    if columns is None:
        columns = 'filename, program, observtn, visit_id, category, template, exp_type, vststart_mjd, visitend_mjd, bstrtime'

    keywords = dict()

    for kw in kwargs:
        val = kwargs[kw]
        if isinstance(kwargs[kw], list):
            # query parameters must be of string type, not ints or other numeric types
            query_val = [str(v) for v in kwargs[kw]]
        elif isinstance(kwargs[kw], dict):
            # some queries, e.g. ranges with min and max, must be specified as a dict, inside a list
            query_val = [kwargs[kw],]
        else:
            # and any scalar parameters should be implicitly wrapped into a list
            query_val = [str(kwargs[kw]),]

        keywords[kw] = query_val

    parameters = {'columns': '*' if all_columns else columns,
                  'filters': set_params(keywords)}
    if verbose:
        print("MAST query parameters:")
        print(parameters)

    responsetable = Mast.service_request(service, parameters)
    responsetable.sort(keys='bstrtime')

    # Add the initial V to visit ID.
    responsetable['visit_id'] = astropy.table.Column(responsetable['visit_id'], dtype=np.dtype('<U12'))

    responsetable.add_column(astropy.table.Column(astropy.time.Time(responsetable['vststart_mjd'], format='mjd').iso, dtype=np.dtype('<U16')),
                            # index=1,
                            name='visit start time')

    return responsetable

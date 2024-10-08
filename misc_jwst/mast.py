
import os

import numpy as np

import astropy, astropy.table
import astropy.units as u

import astroquery
from astroquery.mast import Mast
from astroquery.mast import Observations
import requests

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import misc_jwst.utils

__all__ = ['jwst_keywords_query']

def set_params(parameters):
    """
    Utility function for making dicts used in MAST queries.

    """
    return [{'paramName': p, 'values': v} for p, v in parameters.items()]


def jwst_keywords_query(instrument, columns=None, all_columns=False, verbose=False,  **kwargs):
    """JWST keyword query

    See https://mast.stsci.edu/api/v0/_jwst_inst_keywd.html for keyword field reference
    """
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
    if verbose:
        print(f"Query returned {len(responsetable)} rows")
    if 'bstrtime' in columns:
        responsetable.sort(keys='bstrtime')


    # Some date fields are returned (for database format reasons) as strings, containing 'Date' then an integer
    # giving Unix time in milliseconds. Convert these to astropy times. 
    # This format is not clearly documented, but was reported by MAST archive help desk.
    date_fields = ['date_beg', 'date_end', 'date_obs']   # This is probably not a complete list of which fields to apply this to
    for field_name in date_fields:
        if field_name in responsetable.colnames :
            unix_date_strings = [s[6:-2] for s in responsetable[field_name].value] #  these are strings like '/Date(1679095623534)/'; extract just the numeric part
            times = astropy.time.Time(np.asarray(unix_date_strings, float)/1000, format='unix')
            times.format = 'iso'
            responsetable[field_name] = astropy.table.Column(times)


    # Add the initial V to visit ID.
    if 'visit_id' in columns:
        responsetable['visit_id'] = astropy.table.Column(responsetable['visit_id'], dtype=np.dtype('<U12'))

    if 'vststart_mjd' in columns:
        try:
            responsetable.add_column(astropy.table.Column(astropy.time.Time(responsetable['vststart_mjd'], format='mjd').iso, dtype=np.dtype('<U16')),
                            # index=1,
                            name='visit start time')
        except ValueError:
            pass # For some reason something in that column couldn't be cast to an astropy.Time object

    return responsetable


def visit_which_instrument(visitid):
    """Which instrument did a visit use?

    Note, does not (yet?) handle parallels in a good way. Just returns at most 1 instrument
    """
    program = visitid[1:6]

    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection=["JWST"], proposal_id=[program])
    # Annoyingly, that query interface doesn't return start/end times
    instruments = [val.split('/')[0] for val in set(obs['instrument_name'])]

    # TODO - check more carefully for cases with multiple instruments in parallel
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Because we expect at least some of these may have a warning about no results found
        for inst in instruments:
            if len(jwst_keywords_query(inst, visit_id=visitid[1:])) > 0:
                return inst


def query_visit_time(visitid, verbose=False):
    """ Find start and end time of a visit

    Input: string visit id, like 'V01234005001'
    Returns start and end time as astropy.time.Time instances, or Nones.
    """

    # Get table of times for all visits in that program
    program = visitid[1:6]
    visit_times = query_program_visit_times(program, verbose=verbose)

    #  Scan through the table for a row matching that visit id
    for vid, vstart, vend, inst in visit_times:
        if verbose:
            print(vid, visitid, vid==visitid, inst)
        if vid==visitid:
            # Return times as astropy Time objects
            t0 = astropy.time.Time(vstart, format='mjd')
            t1 = astropy.time.Time(vend, format='mjd')
            t0.format = 'iso'
            t1.format = 'iso'
            return t0, t1
    else:
        return None, None


def get_visit_exposure_times(visitid, extra_columns=""):
    """ Return a table with start and end times for all exposures within a visit"""
    visitid = misc_jwst.utils.get_visitid(visitid)  # handle either input format
    inst = visit_which_instrument(visitid)

    # Set some extra keywords per instrument
    if extra_columns:
        inst_mode_columns = {'NIRCAM': "filter,pupil",
                             'NIRISS': "filter,pupil",
                             'NIRSPEC': 'filter,grating',
                             'MIRI': 'filter,detector,band'}
        extra_query_columns = ", "+inst_mode_columns[inst]
    else:
        extra_query_columns = ""

    res = jwst_keywords_query(inst, visit_id=visitid[1:],
                                       columns = 'filename, visit_id, date_beg_mjd, date_end_mjd, vststart_mjd, visitend_mjd, productLevel, exp_type, effexptm' + extra_query_columns)
    for colname in ['date_beg_mjd', 'date_end_mjd', 'vststart_mjd', 'visitend_mjd']:
        times = astropy.time.Time(res[colname], format='mjd')
        times.format='iso'
        res[colname] = times
    res.sort('date_beg_mjd')

    exps_level2 = res[np.asarray(res['productLevel'].value) != '3']
    # mask out some redundant files
    fns = exps_level2['filename']
    ignore_derived_products_mask = [('s3d' not in fn) and ('x1d' not in fn) and ('i2d' not in fn) for fn in fns]
    exps_level2  = exps_level2[ignore_derived_products_mask]

    if extra_columns:
        keys = inst_mode_columns[inst].split(',')
        exps_level2['optical_elements'] = ["+".join([str(row[k]) for k in keys])  for row in exps_level2]

        # improve handling for MIRI MRS & imager exps, each of which has some missing masked field
        if inst=='MIRI':
            exps_level2['optical_elements'] = [v.replace('+MIRIMAGE+--', '').replace('--+', '') for v in exps_level2['optical_elements']]


    return exps_level2


def query_program_visit_times(program,  verbose=False):
    """ Get the start and end times of all completed visits in a program.

    Parameters
    ----------
    program : int or str
        Program ID
    verbose : bool
        be more verbose in output?

    Returns astropy Table with columns for visit ID and start and end times.
    """

    # use Observations query interface to find all observations
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection=["JWST"], proposal_id=[program])

    # Annoyingly, that query interface doesn't return start/end times,
    # therefore we have to separately call a different query interface for those, per instrument
    instruments = [val.split('/')[0] for val in set(obs['instrument_name'])]
    visit_times = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Because we expect at least some of these may have a warning about no results found
        for inst in instruments:
            if verbose:
                print(f"querying for visits using {inst}")
            visit_times += _query_program_visit_times_by_inst(program, inst)

    # Format outputs
    vids = [v[0] for v in visit_times]
    starts =astropy.time.Time([float(v[1]) for v in visit_times], format='mjd')
    ends = astropy.time.Time([float(v[2]) for v in visit_times], format='mjd')
    insts = [v[3] for v in visit_times]
    vis_table =  astropy.table.Table([vids, starts, ends, insts],
                                     names=('visit_id', 'start_mjd', 'end_mjd', 'instrument'))
    vis_table.sort(keys='start_mjd')
    return vis_table


def _query_program_visit_times_by_inst(program, instrument, verbose=False):
    """ Get the start and end times of all completed visits in a program, per instrument.
    Not intended for general use; this is mostly a helper to query_program_visit_times.

    Getting the vststart_mjd and visitend_mjd fields requires using the instrument keywords
    interface, so one has to specify which instrument ahead of time.

    Parameters
    ----------
    program : int or str
        Program ID
    instrument : str
        instrument name
    verbose : bool
        be more verbose in output?

    returns list of (visitid, start, end) tuples.

    """

    from astroquery.mast import Mast
    svc_table = {'MIRI':'Mast.Jwst.Filtered.Miri',
                 'NIRCAM': 'Mast.Jwst.Filtered.NIRCam',
                 'NIRSPEC': 'Mast.Jwst.Filtered.NIRSpec',
                 'NIRISS': 'Mast.Jwst.Filtered.NIRISS',
                }

    service = svc_table[instrument.upper()]

    collist = 'filename, program, observtn, visit_id, vststart_mjd, visitend_mjd, bstrtime'
    all_columns = False

    def set_params(parameters):
        return [{"paramName" : p, "values" : v} for p, v in parameters.items()]


    keywords = {'program': [str(program),]}
    parameters = {'columns': '*' if all_columns else collist,
                  'filters': set_params(keywords)}

    if verbose:
        print("MAST query parameters:")
        print(parameters)

    responsetable = Mast.service_request(service, parameters)
    if 'bstrtime' in collist:
        responsetable.sort(keys='bstrtime')

    visit_times = []
    for row in responsetable:
        visit_times.append( ('V'+row['visit_id'], row['vststart_mjd'], row['visitend_mjd'], instrument))

    visit_times= set(visit_times)
    return list(visit_times)


def summarize_jwst_observations(targname, radius='30s', exclude_ta=True):
    """Search MAST for JWST obs of a target, in all modes, including upcoming approved planned observations

    Returns a nested dictionary, ordered by instrument_mode, then program ID, then filter

    Parameters
    ----------
    targname : str
        target name, for SIMBAD coords query
    radius : str
        cone search radius, as a string with units parsable by astropy.coordinates
    exclude_ta : bool
        should Target Acquisition exposures be excluded from the results?
    """

    obstable = Observations.query_criteria(
            obs_collection='JWST',
            objectname=targname,
            radius=radius
            )

    has_obs = dict()

    for row in obstable:
        mode = row['instrument_name']
        filters = str(row['filters'])
        proposal = row['proposal_id']
        if exclude_ta and '/TA' in mode:
            continue

        if mode not in has_obs:
            has_obs[mode] = dict()
        if proposal not in has_obs[mode]:
            has_obs[mode][proposal] = set()
        has_obs[mode][proposal].add(filters)

    sorted_results_dict = dict(sorted(has_obs.items()))
    return sorted_results_dict

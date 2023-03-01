import astropy.io.fits as fits

import requests
from bs4 import BeautifulSoup

import os
import pandas

import functools


# Utility functions for working with JWST CRDS and retrieving/displaying metadata about reference files used.

@functools.lru_cache
def retrieve_crds_metadata(ref_filename):
    """Retrieve CRDS database metadata table for a given reference filename.

    Parameters
    ----------
    ref_filename : str
        CRDS reference filename, like 'jwst_nircam_photom_0116.fits'

    Returns a pandas dataframe containing the "Database" table metadata
    for that given file, retrieved from the CRDS web site.

    """
    url = f"https://jwst-crds.stsci.edu/browse/{ref_filename}"

    result = pandas.read_html(url)
    table0 = result[0].rename(columns={0:'keyword', 1:'value'})

    return table0


def describe_crds_file(ref_filename, verbose=False):
    """ Print a convenient human-readable summary of a subset of the CRDS reference file's metadata.

    Parameters
    ----------
    ref_filename : str
        CRDS reference filename, like 'jwst_nircam_photom_0116.fits'
    verbose : bool
        Print out a more lengthy set of metadata?

    """

    table0 = retrieve_crds_metadata(ref_filename)

    if verbose:
        metadata_keys_to_print = ['Pedigree', 'Status', 'Delivery Date', 'Activation Date', 'Useafter Date',
                                  'Descrip', 'Change Level', 'Submit Description']
    else:
        metadata_keys_to_print = ['Pedigree', 'Status', 'Delivery Date', 'Activation Date', 'Descrip']



    for k in metadata_keys_to_print:
        val=table0[table0['keyword']==k]['value'].values[0]
        print(f"   {k:10s}:\t{val} ")


def describe_crds_ref_files_used(filename, verbose=False):
    """Print a convenient human-readable summary of the CRDS ref files used to reduce a given JWST data file.

    Parameters
    ----------
    filename : str
        Some JWST reduced file name, like 'jw02724233001_03104_00002_nrca3_cal.fits.fits'
    verbose : bool
        Print out a more lengthy set of metadata?


    """


    header = fits.getheader(filename)
    ref_keys = [k for k in header.keys() if k.startswith('R_')]

    print(f"== CRDS Reference Files used in preparing {os.path.basename(filename)} ==")
    print(f"   Reduced with CAL_VER  = {header['CAL_VER']}\t\t\tData pipeline version")
    print(f"   Reduced with CRDS_VER = {header['CRDS_VER']}\t\t\tCRDS version")
    print(f"   Reduced with CRDS_CTX = {header['CRDS_CTX']}\t\tCRDS context file")
    print(f"   Reduced on   DATE     = {header['DATE']}\tUTC date file created")
    print("")


    for k in ref_keys:
        i = header.index(k)
        val = header[k]
        if val=='N/A':
            continue
        reffn = val.split('/')[-1]

        print(f"{k:8s}  {header[i-2]:40s}\t {reffn}")
        describe_crds_file(reffn, verbose=verbose)


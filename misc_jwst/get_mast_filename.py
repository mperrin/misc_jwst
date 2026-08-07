#! /usr/bin/env python
# retrieve MAST JWST files using authentication
#
# Sample usage:
# get_mast_filename.py jw01410021001_02101_00001_guider1_uncal.fits
#
# R. White, 2022 February 15

import getopt
import os
import sys

import requests
from tqdm import tqdm


def get_mast_filename(filename, outputdir='.', overwrite=False, skip_duplicates=True, progress=False,
        mast_api_token=None, mast_url="https://mast.stsci.edu/api/v0.1/Download/file"):
    """Download the filename, writing to outputdir

    Default outputdir comes from filename; specify '.' to write to current directory.
    Set overwrite=True to overwrite existing output file.  Default is to raise ValueError.
    Set progress=True to show a progress bar.

    Other parameters are less likely to be useful:
    Default mast_api_token comes from MAST_API_TOKEN environment variable.
    mast_url should be correct unless the API changes.
    """
    if not mast_api_token:
        mast_api_token = os.environ.get('MAST_API_TOKEN')
        if mast_api_token is None:
            raise ValueError("Must define MAST_API_TOKEN env variable or specify mast_api_token parameter")
    assert '/' not in filename, "Filename cannot include directories"
    if outputdir is None:
        outputdir = '_'.join(filename.split('_')[:-1])
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    elif not os.path.isdir(outputdir):
        raise ValueError(f"Output location {outputdir} is not a directory")
    elif not os.access(outputdir, os.W_OK):
        raise ValueError(f"Output directory {outputdir} is not writable")
    outfile = os.path.join(outputdir, filename)

    if os.path.exists(outfile):
        if skip_duplicates:
            print("File already exists; skipping.")
            return
        elif (not overwrite):
            raise ValueError(f"{outfile} exists, not overwritten")



    r = requests.get(mast_url, params={'uri': f"mast:JWST/product/{filename}"},
                headers={'Authorization': f"token {mast_api_token}"}, stream=True)
    r.raise_for_status()

    total_size_in_bytes = int(r.headers.get('content-length', 0))
    block_size = 1024000
    if progress:
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        csize = 0
    with open(outfile, 'wb') as fd:
        for data in r.iter_content(chunk_size=block_size):
            fd.write(data)
            if progress:
                # use the size before uncompression
                dsize = r.raw.tell()-csize
                progress_bar.update(dsize)
                csize += dsize
    if progress:
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")


def usage(msg=None):
    progname = sys.argv[0]
    print(f"""Usage: {progname} [-d output_directory] [-o] [-q] [-h] filename
Retrieve named JWST file from MAST using MAST_API_TOKEN authentication
Example: {progname} jw01410021001_02101_00001_guider1_uncal.fits
Parameters:
-d output directory
   Default is from filename, e.g. jw01410021001_02101_00001_guider1
   Specify -d . to put in current directory
   Directory is created if it does not exist
-o Overwrite existing file (default is not to overwrite)
-q Run in quiet mode (default shows progress bar)
-h Print help info""",file=sys.stderr)
    if msg:
        print(msg,file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:oqh")
    except getopt.error as e:
        usage(str(e))
    progress = True
    overwrite = False
    outputdir = None
    for opt, value in opts:
        if opt == '-d':
            outputdir = value
        elif opt == '-o':
            overwrite = True
        elif opt == '-q':
            progress = False
        elif opt == '-h':
            usage()
        else:
            usage(f"Unknown option '{opt}'")
    # Note this could easily be modified to accept a list of filenames
    if len(args) != 1:
        usage("Specify a single filename")
    filename = args[0]
    get_mast_filename(filename,overwrite=overwrite,progress=progress,outputdir=outputdir)

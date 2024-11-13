import requests
import os
import zipfile
from io import BytesIO
import lxml.etree as etree

import numpy as np

import mirage.apt.read_apt_xml as read_apt_xml

## Globals

aptpath = None

def set_aptpath(path = '/Application`s/APT 2021.2/bin/apt'):
    """Set APT path. This must be done before using any of the functions that invoke APT

    """
    global aptpath
    aptpath = path
    print(f"APT path set to {aptpath}")


## Functions to retrieve and download APT files from STScI

def download_apt_xml(progid):
    r = requests.get('http://www.stsci.edu/jwst/phase2-public/{}.aptx'.format(progid))
    z = zipfile.ZipFile(BytesIO(r.content))
    #fn = z.filelist[1].filename
    retval = z.extract('{}.xml'.format(progid))
    print(retval)


def download_apt_aptx(progid, redownload=False):
    outname = f"{progid}.aptx"
    if not os.path.exists(outname) or redownload:
        r = requests.get('http://www.stsci.edu/jwst/phase2-public/{}.aptx'.format(progid))
        open(outname,"wb").write(r.content)
        print(f"{outname} downloaded")
    else:
        print(f"{outname} file already present.")


def download_apt_aptx_and_xml(progid, redownload=False):
    # Download and save the APT files, and also export their contents to XML
    outname = f"{progid}.aptx"
    if not os.path.exists(outname) or redownload:
        r = requests.get('http://www.stsci.edu/jwst/phase2-public/{}.aptx'.format(progid))
        open(outname,"wb").write(r.content)
        print(f"{outname} downloaded")
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extract('{}.xml'.format(progid))
        print(f"  and extracted {progid}.xml")
    else:
        print(f"{outname} file already present.")

## Functions to parse APTX XML and print outputs

def parse_special_reqs(filename, verbose=False):
    """ Read in special requirement info from an APT XML file

    filename : string
        APT XML filename
    verbose : bool
        print more?

    returns a dict of dicts. Top level dict is indexed by obsID numbers, then
    the inner dicts are the SR names and values for each observation.

    """
    # Mirage doesn't read in the special requirements, so we do that here.

    tree = etree.parse(filename)
    apt = '{http://www.stsci.edu/JWST/APT}'
    observation_data = tree.find(apt+'DataRequests')
    observation_list = observation_data.findall('.//' + apt + 'Observation')

    # Loop through observations, get parameters
    all_srs = {}
    for i_obs, obs in enumerate(observation_list):
        observation_number = np.int(obs.find(apt + 'Number').text)
        if verbose:
            print(observation_number)

        srs = obs.find('.//' + apt +'SpecialRequirements')
        sr_d = {}
        for i_sr, sr in enumerate(srs):
            sr_d[sr.tag[len(apt):]] = sr.text
        if verbose:
            print(sr_d)
        all_srs[observation_number] = sr_d
    return all_srs

def describe_setup(results, i, sr_info, progid):
    """ Convenience helper for printing the handful of parameters we're interested in.
    Quick and dirty; kind of a kludge.

    results : output from read_xml
    i : index of observation
    sr_info : output from parse_special_reqs
    progid : APT program ID number

    returns a string.

    """
    obsid = int(results['ObservationID'][i])
    guidemode = sr_info[obsid].get('PcsMode','FineGuide')

    if results['Instrument'][i]=='NIRCAM':
        template = results['APTTemplate'][i]
        # TODO customize what gets printed per each WFSC template?
        retval = template+", "+results['ShortFilter'][i]+", NGRP="+results['Groups'][i]
    else:
        retval =  results['APTTemplate'][i]

    filenameout = "jw{:05d}{:03d}001_*".format(progid, obsid)

    retval= "{:45s}\tGuiding={:10s}\t{}".format(retval, guidemode, filenameout)
    return retval


def summarize_program(progid):
    """ Read in an APT XML file, prepare text file of useful info summary

    progid : APT program ID, e.g. 1158
    """
    print(progid)

    # Read in the APT XML, using MIRAGE's code for that.
    reader = read_apt_xml.ReadAPTXML()

    xmlfn = f'{progid}.xml'
    if not os.path.exists(xmlfn):
        download_apt_xml(progid)

    try:
        res = reader.read_xml(xmlfn, verbose=False)
    except ValueError:
        return "Unsupported template type for {}.xml".format(progid)
    if len(res['Title'])==0:
        return "Problem reading {}.xml".format(progid)

    # Also read the special requirements info, which MIRAGE doesn't yet do
    sr_info = parse_special_reqs('{}.xml'.format(progid), verbose=False)

    output = ""
    # Record the program title
    output+="\nAPT {}:  {}\n\n".format(progid, res['Title'][0] )

    # Generate descriptions of all observations. These will typically be repeated many times,
    # because there are usually multiple exposures per observation.
    labels = ['Obs {}: {}'.format(num, label) for num, label in zip (res['ObservationID'], res['ObservationName']) ]

    # so now we iterate through, skip duplicates, and output one line per observation
    prev = ""
    for i, label in enumerate(labels):
        if label == prev:
            continue
        prev=label
        output+="\t{:70s}\t{}\n".format(label, describe_setup(res, i, sr_info, progid))

    return output


def get_program_obslabels(progid):
    """ Read in an APT XML file, prepare text file of useful info summary

    progid : APT program ID, e.g. 1158
    """
    print(progid)

    xmlfn = f'{progid}.xml'
    if not os.path.exists(xmlfn):
        download_apt_xml(progid)

    # Read in the APT XML, using MIRAGE's code for that.
    reader = read_apt_xml.ReadAPTXML()
    try:
        res = reader.read_xml(xmlfn, verbose=False)
    except ValueError:
        return "Unsupported template type for {}.xml".format(progid)
    if len(res['Title'])==0:
        return "Problem reading {}.xml".format(progid)

    # Also read the special requirements info, which MIRAGE doesn't yet do
    #sr_info = parse_special_reqs('{}.xml'.format(progid), verbose=False)

    obslabels = dict()
    obstemplates = dict()
    for num, label, template in zip (res['ObservationID'], res['ObservationName'], res['APTTemplate']):
        obslabels[num] = label
        obstemplates[num] = template
    return obslabels, obstemplates





## Functions to invoke APT and calculate some outputs

def export_timing_json(prog_id, redo=False):
    # Invoke APT command line to export timing report.
    # Thanks to Andrew Myers on APT team for instruction

    if aptpath is None:
        raise RuntimeError("You must use set_aptpath before you can call this function!")

    if not os.path.exists(f'{prog_id}.timing.json') or redo:
        print(f"Exporting timing summary for {prog_id}")
        import subprocess
        subprocess.run([aptpath, '--nogui', '-export', 'timing.json', f'{prog_id}.aptx'])

def timing_summary_from_json(progid):
    # Read info from the APT exported .timing.json files and display summary
    fn = f'{progid}.timing.json'
    import json

    print(f"Timing summary for APT {progid}")
    with open(fn) as file:
        timedata = json.load(file)
    print(f"Total charged in APT for program {progid}: {timedata['charged_time']} hours")

    tot_charged = 0

    for i, obs in enumerate(timedata['observations']):
        print(f'Obs {obs["id"]:2}: Charged {obs["charged_duration_seconds"]:4} s\t({obs["template"]}, {obs["number_of_visits"]} visit)')
        tot_charged += obs["charged_duration_seconds"]
    print(f"\tSum of observations in APT for program {progid}: {tot_charged} s = \t{tot_charged/3600:.02f} hours\n")

def get_obs_description(res, desired_obsid):
    # helper function for the awkward task of getting observation description from the parsed XML
    try:
        return [obsname for obsid,obsname in zip(res['ObservationID'], res['ObservationName']) if int(obsid) == desired_obsid][0]
    except:
        return ""

def timing_summary_from_json_to_excel(progid, workbook_in=None, include_descriptions=True):
    """ You can use this if you need to customize which observations are included in your sums, for instance if you have some visitions that are on hold or contingency and should not be included. This outputs Excel files. Edit the spreadsheet and set the "Include" column to 'N' for observations to skip from the sums
    """
    import xlsxwriter
    from astropy.table import Table
    fn = f'{progid}.timing.json'
    # Create a workbook and add a worksheet.
    if workbook_in is not None:
        workbook = workbook_in
    else:
        workbook = xlsxwriter.Workbook(f'timing_{progid}.xlsx')
    worksheet = workbook.add_worksheet(str(progid))
    bold = workbook.add_format({'bold': True})


    worksheet.write(0, 0, "Timing summary for APT:", bold)
    worksheet.write(0, 2, progid, bold)
    for i, header in enumerate(['Obs', 'Total Charged (s)', 'Template', 'Include in sum?']):
        worksheet.write(1,i,header)

    with open(fn) as file:
        timedata = json.load(file)

    nobs = len(timedata['observations'])

    print(f'--- {prog_id}')
    if include_descriptions and prog_id not in skip_programs:
        import mirage.apt.read_apt_xml as read_apt_xml
        reader = read_apt_xml.ReadAPTXML()
        res = reader.read_xml(f'{progid}.xml', verbose=False)

    for i, obs in enumerate(timedata['observations']):
        worksheet.write(i+2, 0, obs["id"])
        worksheet.write(i+2, 1, obs["charged_duration_seconds"])
        worksheet.write(i+2, 2, obs["template"])
        worksheet.write(i+2, 3, 'Y')
        if include_descriptions and prog_id not in skip_programs:
            worksheet.write(i+2, 4, get_obs_description(res, obs["id"]))


    worksheet.write(nobs+3, 0, 'Sum:')
    worksheet.write(nobs+3, 1, f'=sumif(D2:D{nobs+2}, "Y", B2:B{nobs+2})')
    worksheet.write(nobs+3, 2, 'seconds, for')
    worksheet.write(nobs+3, 3, f'=countif(D2:D{nobs+2}, "Y")')
    worksheet.write(nobs+3, 4, 'observations')

    worksheet.write(nobs+4, 0, 'Sum:')
    worksheet.write(nobs+4, 1, f'=sumif(D2:D{nobs+2}, "Y", B2:B{nobs+2})/3600')
    worksheet.write(nobs+4, 2, 'hours, for')
    worksheet.write(nobs+4, 3, f'=countif(D2:D{nobs+2}, "Y")')
    worksheet.write(nobs+4, 4, 'observations')

    # adjust some formatting like widths
    worksheet.set_column('B:B', 14)
    worksheet.set_column('C:C', 40)


    if workbook_in is None:
        workbook.close()
        print(f"Saved to timing_{progid}.xlsx")

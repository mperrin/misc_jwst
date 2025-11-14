# Misc utility and wrapper functions for running the JWST pipeline conveniently


import os, glob
import warnings

import astropy.io.fits as fits

import stcal
import jwst
from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_spec2, calwebb_image3, calwebb_spec3
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase


# The optimal settings will vary depending on dataset and details of analysis & science goals...
# The following just intended as suggestions and/or reminders of parameter names which are useful to tweak

recommended_l1_param_overrides = {
                        'jump': {'maximum_cores': 'all',         # Parallelize 
                                },
                        'ramp_fit': {'maximum_cores': 'all',        # Parallelize 
                                    'suppress_one_group': False,      # Suppress fit when 0th group is unsaturated
                                   },
                        'saturation': {'n_pix_grow_sat': 0,},      # Expand/pad saturation around saturated pixels?
                        'persistence': {'skip': True},
                      }

recommended_l2_param_overrides = {
                        'resample': {'skip': True,         # Don't make i2d files we don't need.
                    }}


def do_stage1(filenamelist, overwrite=False, output_dir='.', l1_param_overrides=None):
    """Run pipeline stage 1, with some wrapper customizations, for a list of files

    Returns list of output files
    """
    outputs = []

    for filename in filenamelist:
        outputs.append(do_one_stage1(filename, overwrite=overwrite, output_dir=output_dir, l1_param_overrides=l1_param_overrides))
    return outputs



def do_one_stage1(inputfilename, overwrite=False, output_dir='.', l1_param_overrides=None):
    """Run pipeline stage 1, with some wrapper customizations for one file

    * Check if product already exists, and don't overwrite if so.
    * See above parameter overrides

    """

    outfile = os.path.join(output_dir, os.path.basename(inputfilename).replace("_uncal.fits", "_rate.fits"))

    if os.path.exists(outfile) and not overwrite:
        print(f"Already exists: {outfile}")
        return outfile

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    detector1 = calwebb_detector1.Detector1Pipeline()

    # Run pipeline
    print(f"Reducing {inputfilename} to {outfile}")
    detector1.call(inputfilename, output_dir=output_dir, save_results=True,
                   steps = l1_param_overrides)
    return outfile


def do_stage2image(filenamelist, overwrite=False, output_dir='.', l2_param_overrides=None, spec=False, bgfiles=None):
    """Run pipeline stage 2, with some wrapper customizations, for a list of files

    Returns list of output files
    """
    outputs = []

    for filename in filenamelist:
        outputs.append(do_one_stage2(filename, overwrite=overwrite, output_dir=output_dir,
                                     l2_param_overrides=l2_param_overrides, spec=spec, bgfiles=bgfiles))
    return outputs

#def do_stage2spec(filenamelist, overwrite=False, output_dir='.', l2_param_overrides=None, spec=True):
#    """Run pipeline stage 2, with some wrapper customizations, for a list of files
#
#    Returns list of output files
#
#    This is SAME AS stage2image, just flips the spec keyword to true!
#    Intentional duplciation to make it easier to find this
#    """
#    outputs = []
#
#    for filename in filenamelist:
#        outputs.append(do_one_stage2(filename, overwrite=overwrite, output_dir=output_dir, l2_param_overrides=l2_param_overrides, spec=spec))
#    return outputs
#

def do_one_stage2(inputfilename, overwrite=False, output_dir='.', spec=False, l2_param_overrides=None,
                  bgfiles=None):
    """Run pipeline stage 2, with some parameter customizations

    * Check if product already exists, and don't overwrite if so.
    * See above parameter overrides.

    """

    if l2_param_overrides is None:
        l2_param_overrides = dict()

    outfile = os.path.join(output_dir, os.path.basename(inputfilename).replace("_rate.fits", "_cal.fits"))

    if os.path.exists(outfile) and not overwrite:
        print(f"Already exists: {outfile}")
        return outfile

    if spec:
        stage2 = calwebb_spec2.Spec2Pipeline()
    else:
        stage2 = calwebb_image2.Image2Pipeline()

    if bgfiles is not None:
        # User has provided a list of background files for stage 2 background subtraction
        # In this case we need to make an association file
        asnfilename = os.path.join(output_dir, 'l2asn.json')
        make_l2_association(inputfilename, bgfiles, asnfilename)
        pipeline_input = asnfilename

    else:
        pipeline_input = inputfilename

    # Run pipeline
    print(f"Reducing {inputfilename} to {outfile}")
    stage2.call(pipeline_input, output_dir=output_dir, save_results=True,
                   steps = l2_param_overrides)
    return outfile



def remove_1f_cw(cal2file, splitamps=True, overwrite=False, vertical=False, plot=False):
    """Remove 1/f noise via Chris Willott's method
    See https://github.com/chriswillott/jwst/blob/master/image1overf.py

    Parameters
    ----------
    splitamps : bool
        Treat each amplifier separately. Using this works well sometimes but not always;
        better on on sparse fields.
    """

    from image1overf import sub1fimaging


    cal21overffile = cal2file.replace('_cal.fits','_cal1f.fits')
    print ('Running 1/f correction on {} to produce {}'.format(cal2file,cal21overffile))

    if os.path.exists(cal21overffile) and not overwrite:
        print(f"Already exists: {cal21overffile}")
        return

    with fits.open(cal2file) as cal2hdulist:
        if cal2hdulist['PRIMARY'].header['SUBARRAY']=='FULL' or cal2hdulist['PRIMARY'].header['SUBARRAY']=='SUB256':
            sigma_bgmask=3.0
            sigma_1fmask=2.0


            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Ignore astropy warning about NaNs being present; it's ok
                correcteddata = sub1fimaging(cal2hdulist,sigma_bgmask, sigma_1fmask, splitamps)


            if plot:
                predata = cal2hdulist['SCI'].data[4:2044,4:2044]
                dcoffset = np.nanmedian(predata)
                fig, axes = plt.subplots(figsize=(16,9),
                                         ncols=3)
                vm = 0.03
                norm = matplotlib.colors.Normalize(vmin=-vm, vmax=vm)
                axes[0].imshow(predata - dcoffset, norm=norm)
                axes[0].set_title(cal2file)
                axes[2].imshow(correcteddata - dcoffset, norm=norm)
                axes[2].set_title("Corrected")
                axes[1].imshow(predata - correcteddata , norm=norm)
                axes[1].set_title("1/f noise model")


                plotfn = f"plot_remove_1f_{os.path.basename(cal2file)[:-5]}.pdf"
                plt.tight_layout()
                plt.savefig(plotfn)
                print(f" plot saved to {plotfn}")


            if cal2hdulist['PRIMARY'].header['SUBARRAY']=='FULL':
                cal2hdulist['SCI'].data[4:2044,4:2044] = correcteddata
            elif cal2hdulist['PRIMARY'].header['SUBARRAY']=='SUB256':
                cal2hdulist['SCI'].data[:252,:252] = correcteddata
            cal2hdulist[0].header['ONEOVERF'] = ("COMPLETED", '1/f noise removal using sub1fimaging.py')
            cal2hdulist[0].header['ONEF_SA'] = (splitamps, 'Split amps for 1/f noise removal ?')
            cal2hdulist[0].header['ONEFVERT'] = (splitamps, 'Also include vertical stripe removal?')


            cal2hdulist.writeto(cal21overffile, overwrite=True)



def get_instrument_filter(model):
    """ Get instrument filter, including if it's in a pupil wheel
    """

    inst = model.meta.instrument.name
    filt = model.meta.instrument.filter
    if inst == 'NIRSPEC':
        filt = model.meta.instrument.grating + "-" + model.meta.instrument.filter
    if inst == 'NIRISS' and filt == 'CLEAR':
        filt = model.meta.instrument.pupil
    return filt


def make_l3_association(filelist, asn_description=None, spec=False):
    """Make an association file from a list of input files """
    from jwst.associations import asn_from_list
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


    with jwst.datamodels.open(filelist[0]) as model:
        inst = model.meta.instrument.name
        filt = get_instrument_filter(model)
        progid = model.meta.observation.program_number
        obs = model.meta.observation.observation_number

    optional_description = '' if asn_description is None else '_'+asn_description

    asn_name = f'jw{progid}_o{obs}_{inst.lower()}_{filt.lower()}{optional_description}.json'

    if spec:
        if inst=='NIRSpec':
            # note, 'filt' will contain both filter and grating concatenated
            product_name = f'jw{progid}-{obs}_{{source_id}}_{inst}_{filt}-{{slit_name}}'
        else:
            # MIRI
            product_name = f'jw{progid}-{obs}_{inst}_{{slit_name}}'
    else:
        product_name = asn_name

    # Using absolute paths avoids an annoying warning message from asn_from_list
    import os.path
    abs_filelist = [os.path.abspath(f) for f in filelist]

    act_asn = asn_from_list.asn_from_list(abs_filelist,
                                          rule=DMS_Level3_Base,
                                          product_name=product_name)

    if spec:
        act_asn['asn_type'] = 'spec3'
    else:
        act_asn['asn_type'] = 'image3'
        act_asn['asn_rule'] = 'candidate_asn_lv3image'

    act_asn['program'] = progid

    with open(asn_name, 'w') as outfile:
        name, serialized = act_asn.dump(format='json')
        outfile.write(serialized)

    return asn_name

def make_l2_association(onescifile, bgfiles, asnfile, prodname='level2'):
    """ Make L2 association file
    Adapted from https://github.com/spacetelescope/jwst-pipeline-notebooks/blob/main/notebooks/MIRI/Imaging/JWPipeNB-MIRI-imaging.ipynb
    
    Includes check for matching of filters, from code originally for MIRI


    """

    # Define the basic association of science files
    asn = asn_from_list.asn_from_list([onescifile], rule=DMSLevel2bBase, product_name=prodname)  # Wrap in array since input was single exposure

    # Filter configuration for this sci file
    with fits.open(onescifile) as hdu:
        hdu.verify()
        hdr = hdu[0].header
        this_filter = hdr['FILTER']

    if hdu[0].header['INSTRUME'] != 'MIRI':
        raise NotImplementedError('this function needs to be tested and enhanced for non-MIRI usage, w.r.t FILTER keyword matching')

    # If backgrounds were provided, find which are appropriate to this
    # filter and add to association
    for file in bgfiles:
        with fits.open(file) as hdu:
            hdu.verify()
            if (hdu[0].header['FILTER'] == this_filter):
                asn['products'][0]['members'].append({'expname': file, 'exptype': 'background'})              

    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)


def create_l3_asns_grouped_by_filter(cal_files, asn_description=None):
    """ Create level 3 associations, one filter at a time
    """
    import collections
    filter_sets = collections.defaultdict(list)

    for fn in cal_files:
        with jwst.datamodels.open(fn) as model:
            print(f"{fn} has filter = {model.meta.instrument.filter}")
            filter_sets[model.meta.instrument.filter].append(fn)
    l3_asns = []
    for filt in filter_sets:
        filt_asn = make_l3_association(filter_sets[filt], asn_description=asn_description)
        l3_asns.append(filt_asn)
        print(f"Created {filt_asn} with {len(filter_sets[filt])} files")

    return l3_asns


def do_stage3image(cal_files=None, association_file=None, l3_param_overrides=None,
                   output_dir=".", overwrite=False,
                   cleanup_crfs=True, asn_description=None):
    """ Run stage3 processing. This may **either** be called with an association file, or a list of cal files

    """

    if l3_param_overrides is None:
        l3_param_overrides = {}

    if association_file is not None:
        associations = [association_file,]
    else:
        # Create associations for the supplied input files, after grouping them by filters
        associations = create_l3_asns_grouped_by_filter(cal_files, asn_description=asn_description)


    i2d_files = []
    for asn_file in associations:
        outfile = os.path.join(output_dir, asn_file.replace('.json', '_i2d.fits'))
        if os.path.exists(outfile) and not overwrite:
            print(f"Already exists: {outfile}")
        else:
            image3 = calwebb_image3.Image3Pipeline()
            image3.call(asn_file,
                        steps=l3_param_overrides,
                        output_dir=output_dir,
                        save_results=True)
        i2d_files.append(outfile)

    if cleanup_crfs:
        crf_files = glob.glob(os.path.join(output_dir, '*crf.fits'))
        if len(crf_files) > 0:
            print(f"Cleaning up {len(crf_files)} *_crf.fits intermediate data files")
            for fn in crf_files:
                os.remove(fn)

    return i2d_files


def do_stage3spec(association_file, l3_param_overrides=None):
    if l3_param_overrides is None:
        l3_param_overrides = {}

    spec3 = calwebb_spec3.Spec3Pipeline()
    spec3.call(association_file,
                steps=l3_param_overrides,
                save_results=True)

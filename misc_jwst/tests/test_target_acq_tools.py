import misc_jwst.target_acq_tools

# NIRCam 

def test_nircam_coron_ta():
    misc_jwst.target_acq_tools.nrc_ta_analysis('6005:30:1')

    # try a different input format, and try different parameters
    misc_jwst.target_acq_tools.nrc_ta_analysis('V04454003001', return_pointing_offsets=True, verbose=False, plot=False)

def test_nircam_wfsc_ta():
    misc_jwst.target_acq_tools.nrc_ta_analysis('6698:182:1')


def test_nircam_tso_ta():
    misc_jwst.target_acq_tools.nrc_ta_analysis('V01068006001')

# NIRISS

def test_niriss_ami_ta():
    misc_jwst.target_acq_tools.nis_ta_analysis('V01260103001')


def test_niriss_soss_ta():
    misc_jwst.target_acq_tools.nis_ta_analysis('V06606024001')


# MIRI

def test_miri_lrs_ta():
    misc_jwst.target_acq_tools.miri_ta_analysis('V01536027001')


def test_miri_lrs_slitless_ta():
    misc_jwst.target_acq_tools.miri_ta_analysis('V01529005001')


def test_miri_lrs_ta():
    misc_jwst.target_acq_tools.miri_ta_analysis('V05299011001')



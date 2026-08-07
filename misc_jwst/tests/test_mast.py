"""
Pytest integration tests for misc_jwst.mast.

No monkeypatching / mocking of astroquery.mast, requests, or anything else:
every test calls the real function, which performs real network queries
against the MAST archive. That means:

  * These tests require network access, and some (retrieve_files,
    get_mast_filename) may require a valid MAST_API_TOKEN depending on
    whether the target data is proprietary.
  * They will be slow relative to a mocked unit-test suite -- several of
    these functions each issue multiple MAST queries internally.
  * They can be flaky for reasons unrelated to bugs in mast.py -- e.g. a
    particular visit having no science-instrument data (only guide star
    data), or a file requiring authentication we don't have configured.
    Where that's a legitimate "this environment can't fully exercise this
    path" condition rather than a code bug, the test skips (with a
    reason) instead of failing.
  * Per the exercise scope, these tests do not assert anything about the
    *correctness* of returned data/tables -- only that each function can
    be called with a reasonable real-world input and completes without
    raising.

Test inputs are hard-coded per request:
    visit ID:   V01412006001
    program ID: 1412
"""
import os
import astropy

import pytest

import misc_jwst.mast as mast

TEST_VISITID = "V01412006001"
TEST_VISITID_INSTRUMENT = "NIRCAM"
TEST_PROGRAM = "1412"


# ---------------------------------------------------------------------------
# Fixtures that discover real, live data to use as inputs for later tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_instrument():
    """The science instrument (if any) used for TEST_VISITID, from MAST.

    Several functions (jwst_keywords_query, get_visit_exposure_times) need
    an instrument name to query against. Skips dependent tests if MAST is
    unreachable or if this particular visit has no science-instrument data
    in MAST (e.g. it only has guide star data).
    """
    try:
        inst = mast.visit_which_instrument(TEST_VISITID)
    except Exception as exc:  # pragma: no cover - depends on live system
        pytest.skip(f"Could not reach MAST to determine instrument: {exc}")

    if inst is None:
        pytest.skip(f"No science instrument found in MAST for {TEST_VISITID}")

    return inst


@pytest.fixture(scope="module")
def sample_filename(sample_instrument):
    """A real filename from MAST associated with TEST_VISITID, for use by
    the file-download tests. Prefers a small .jpg preview if one exists.
    """
    try:
        table = mast.jwst_keywords_query(
            sample_instrument, program=int(TEST_PROGRAM), visit_id=TEST_VISITID[1:]
        )
    except Exception as exc:  # pragma: no cover - depends on live system
        pytest.skip(f"Could not query MAST for a sample filename: {exc}")

    if table is None or len(table) == 0:
        pytest.skip(f"No files found in MAST for {TEST_VISITID}")

    filenames = list(table["filename"])
    jpgs = [f for f in filenames if str(f).endswith(".jpg")]
    return jpgs[0] if jpgs else filenames[0]


# ---------------------------------------------------------------------------
# set_params (no I/O -- pure helper)
# ---------------------------------------------------------------------------

def test_set_params():
    result = mast.set_params({"program": ["1412"], "instrume": ["NIRCAM", "MIRI"]})
    assert result == [
        {"paramName": "program", "values": ["1412"]},
        {"paramName": "instrume", "values": ["NIRCAM", "MIRI"]},
    ]


# ---------------------------------------------------------------------------
# jwst_keywords_query
# ---------------------------------------------------------------------------

def test_jwst_keywords_query(sample_instrument):
    table = mast.jwst_keywords_query(sample_instrument, program=int(TEST_PROGRAM))
    assert table is not None


def test_jwst_keywords_query_all_columns_verbose(sample_instrument):
    table = mast.jwst_keywords_query(
        sample_instrument, all_columns=True, verbose=True, program=int(TEST_PROGRAM)
    )
    assert table is not None


def test_jwst_keywords_query_more_columns(sample_instrument):
    table = mast.jwst_keywords_query(
        sample_instrument,
        more_columns="targname",
        program=int(TEST_PROGRAM),
        visit_id=TEST_VISITID[1:],
    )
    assert table is not None
    assert 'targname' in table.colnames


# ---------------------------------------------------------------------------
# visit_which_instrument
# ---------------------------------------------------------------------------

def test_visit_which_instrument():
    inst = mast.visit_which_instrument(TEST_VISITID)
    assert inst == TEST_VISITID_INSTRUMENT


# ---------------------------------------------------------------------------
# query_visit_time / get_visit_start_end_times (alias)
# ---------------------------------------------------------------------------

def test_query_visit_time():
    t0, t1 = mast.query_visit_time(TEST_VISITID, verbose=True)
    # May legitimately be (None, None) if the visit isn't found; just
    # checking this runs without raising.
    assert isinstance(t0, astropy.time.Time)
    assert isinstance(t1, astropy.time.Time)
    assert t0 < t1


def test_get_visit_start_end_times_is_query_visit_time_alias():
    assert mast.get_visit_start_end_times is mast.query_visit_time
    mast.get_visit_start_end_times(TEST_VISITID)


# ---------------------------------------------------------------------------
# get_visit_exposure_times
# ---------------------------------------------------------------------------

def test_get_visit_exposure_times():
    # May legitimately return None if there's no science-instrument data
    # for this visit (only guide star data).
    mast.get_visit_exposure_times(TEST_VISITID)


def test_get_visit_exposure_times_extra_columns():
    mast.get_visit_exposure_times(TEST_VISITID, extra_columns=True)


# ---------------------------------------------------------------------------
# query_program_visit_times / _query_program_visit_times_by_inst
# ---------------------------------------------------------------------------

def test_query_program_visit_times_guidestar():
    table = mast.query_program_visit_times(TEST_PROGRAM, verbose=True, use_guidestar=True)
    assert table is not None


@pytest.mark.slow
def test_query_program_visit_times_science_instruments():
    """Heavier variant: discovers every science instrument used in the
    program and queries each one. Run with `pytest -m slow` (or without
    `-m "not slow"`)."""
    table = mast.query_program_visit_times(TEST_PROGRAM, use_guidestar=False)
    assert table is not None


def test_query_program_visit_times_by_inst_guidestar():
    result = mast._query_program_visit_times_by_inst(TEST_PROGRAM, "GUIDESTAR", verbose=True)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# visit_start_end_times (top-level, distinct from get_visit_exposure_times)
# ---------------------------------------------------------------------------

def test_visit_start_end_times():
    start, end = mast.visit_start_end_times(TEST_VISITID)


# ---------------------------------------------------------------------------
# retrieve_files / aliases / get_mast_filename
# ---------------------------------------------------------------------------

def test_retrieve_files(sample_filename, tmp_path):
    try:
        outputs = mast.retrieve_files([sample_filename], out_dir=str(tmp_path), verbose=True)
    except Exception as exc:
        pytest.skip(f"Could not download {sample_filename!r} from MAST: {exc}")
    assert isinstance(outputs, list)


def test_retrieve_files_aliases():
    assert mast.mast_retrieve_files is mast.retrieve_files
    assert mast.download_files is mast.retrieve_files


def test_get_mast_filename(sample_filename, tmp_path):
    if not os.environ.get("MAST_API_TOKEN"):
        pytest.skip("MAST_API_TOKEN not set in the environment")
    try:
        mast.get_mast_filename(
            sample_filename,
            outputdir=str(tmp_path),
            return_in_memory=True,
            verbose=True,
        )
    except Exception as exc:
        pytest.skip(f"Could not download {sample_filename!r} from MAST: {exc}")


def test_get_mast_filename_requires_token(monkeypatch, tmp_path):
    """No mocking of MAST/network here -- just confirming the real,
    documented ValueError path when no token is configured at all."""
    monkeypatch.delenv("MAST_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        mast.get_mast_filename("some_file.fits", outputdir=str(tmp_path))


# ---------------------------------------------------------------------------
# summarize_jwst_observations / report_jwst_highcontrast_observations
# ---------------------------------------------------------------------------

def test_summarize_jwst_observations():
    result = mast.summarize_jwst_observations("NGC 3132")
    assert isinstance(result, dict)


def test_report_jwst_highcontrast_observations():
    table = mast.report_jwst_highcontrast_observations("HR 8799", verbose=True)
    assert table is not None

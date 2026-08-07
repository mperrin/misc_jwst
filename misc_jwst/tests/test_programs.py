import misc_jwst.programs
import astropy.table

def test_scrape_go_programs():
    tables = misc_jwst.programs.scrape_accepted_go_programs(cycle=4, label_by_header=True)

    assert len(tables) > 1
    for label, tab in tables.items():
        assert isinstance(tab, astropy.table.Table)

def test_scrape_gto_programs():
    table = misc_jwst.programs.scrape_accepted_gto_programs(cycle=1)

    assert len(table) > 1
    assert isinstance(table, astropy.table.Table)



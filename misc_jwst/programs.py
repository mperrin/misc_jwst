import numpy as np
import pandas as pd
import requests
import io
import astropy.table


### Functions for accessing lists of accepted programs from STScI's public web pages

import astropy.time, astropy.units as u
_CURRENT_CYCLE = int(np.ceil((astropy.time.Time.now()-astropy.time.Time('2022-07-01')).to_value(u.year)))

def scrape_accepted_go_programs(cycle=1,
                             split_hours=True,
                             verbose=True,
                             ):
    from bs4 import BeautifulSoup
    category='go'
    url = f"https://www.stsci.edu/jwst/science-execution/approved-programs/general-observers/cycle-{cycle}-{category}"
    if verbose:
        print(f"Retrieving {url}")
    req = requests.get(url, )
    soup = BeautifulSoup(req.content, 'html.parser')

    tables = []
    for table in soup.findAll("table"):
        if table.findParent("table") is None:
            tables.append(pd.read_html(  io.StringIO(str(table)) )[0])
    df = pd.concat(tables)
    if verbose:
        print(f"Found {len(tables)} tables of programs")
        print (len(df))
        print(f"Found {len(df)} total observing programs")

    #print(f"Returning selected table(s) {tables_to_include}")
    return_tables = []
    for index in range(len(tables)): # tables_to_include:
        # Convert pandas to astropy
        aptable = astropy.table.Table.from_pandas(tables[index])
        if split_hours:
            if cycle == 3:
                hourskey = 'Prime/Parallel Time (hours)'
            else:
                hourskey = 'Prime/ Parallel Time (hours)'
            hours = aptable[hourskey]
            prime_hours = []
            parallel_hours = []
            for h in hours:
                #print(h)
                if '/' in str(h):
                    prime, parallel = h.split('/')
                elif str(h) == '--': # some cycle 1 programs have this
                    prime, parallel = 0,0
                else:
                    prime, parallel = h, 0
                prime_hours.append(prime)
                parallel_hours.append(parallel)
            prime_hours = np.asarray(prime_hours, float)
            parallel_hours = np.asarray(parallel_hours, float)
            aptable['Prime_hours'] = prime_hours
            aptable['Parallel_hours'] = parallel_hours
        if verbose:
            print(f"Table {index+1} has {len(aptable)} programs")
   
        return_tables.append(aptable)
    return return_tables


def scrape_accepted_gto_programs(cycle=1,
                             split_hours=True,
                             verbose=True,
                             ):
    from bs4 import BeautifulSoup
    url = f"https://www.stsci.edu/jwst/science-execution/approved-programs/guaranteed-time-observations"
    if verbose:

        print(f"Retrieving {url}")
    req = requests.get(url, )
    soup = BeautifulSoup(req.content, 'html.parser')

    tables = []
    for table in soup.findAll("table"):
        if table.findParent("table") is None:
            tables.append(pd.read_html(  io.StringIO(str(table)) )[0])
    df = pd.concat(tables)

    if verbose:
        print(f"Found {len(tables)} tables of programs")
        print (len(df))
        print(f"Found {len(df)} total observing programs")

    #print(f"Returning selected table(s) {tables_to_include}")
    return_tables = []

    tables_to_include = [3-cycle,]  # This is hard coded for cycles 3, 2,1 on that page

    for index in tables_to_include:
        # Convert pandas to astropy
        aptable = astropy.table.Table.from_pandas(tables[index])
        if split_hours:
            hourskey = 'Allocated Hours'
            hours = aptable[hourskey]
            prime_hours = []
            parallel_hours = []
            for h in hours:
                #print(h)
                if '/' in str(h):
                    prime, parallel = h.split('/')
                elif str(h) == '--': # some cycle 1 programs have this
                    prime, parallel = 0,0
                else:
                    prime, parallel = h, 0
                prime_hours.append(prime)
                parallel_hours.append(parallel)
            prime_hours = np.asarray(prime_hours, float)
            parallel_hours = np.asarray(parallel_hours, float)
            aptable['Prime_hours'] = prime_hours
            aptable['Parallel_hours'] = parallel_hours
        if verbose:
            print(f"Table {index+1} has {len(aptable)} programs")
   
        return_tables.append(aptable)
    return return_tables[0]


def hours_by_cycle():
    from bs4 import BeautifulSoup

    go_programs = dict()

    for cycle in range(1,_CURRENT_CYCLE+1):

        tab_cy= scrape_accepted_go_programs(cycle=cycle, verbose=False)
        go_programs[cycle] = tab_cy

        hours = np.sum([t['Prime_hours'].sum() for t in tab_cy])
        print(f"GO Cycle {cycle}:\t{hours:.1f}")

    gto_programs = dict()

    for cycle in [1,2,3]:

        tab_cy= scrape_accepted_gto_programs(cycle=cycle, verbose=False)
        gto_programs[cycle] = tab_cy

        hours = np.sum([t['Prime_hours'].sum() for t in tab_cy])
        print(f"GTO Cycle {cycle}:\t{hours:.1f}")

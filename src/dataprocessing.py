"""
Jacob McKenney & Luke Sala

Imports, cleans, and joins data from multiple datasets to be used
for analysis of tornadoes withiin the United States from 1950 - 2020.
Datasets include:
    - Tornado info: 'data/1950-2020_all_tornadoes.csv'
    - State info & geometries: 'data/gz_2010_us_040_00_5m.json'
    - Census information by county: Acquired via API calls
"""
from census import Census
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


STATE_FILE = '../data/gz_2010_us_040_00_5m.json'
API_KEY = 'd3687e22fcd51ce480482a5caa07e0ae239c77a5'


def retrieve_census_data(c: Census, year) -> pd.DataFrame:
    """
    Takes in an instantiated census object from the Census library along with
    a year and retrieves the pertinent data from the U.S. government via
    API. The following data is returned in a dataframe:
    - NAME: County name
    - B19013_001E: Median household income in the past 12 months
      (2020 inflation-adjusted dollars)
    - C17002_001E: count of ratio of income to poverty in the
      past 12 months (total)
    - C17002_002E: count of ratio of income to poverty in the past
      12 months (< 0.50)
    - C17002_003E: count of ratio of income to poverty in the past
      12 months (0.50 - 0.99)
    - B01003_001E: Total population
    """
    wa_census = c.acs5.state_county(
        fields=('NAME', 'B19013_001E', 'C17002_001E', 'C17002_002E',
                'C17002_003E', 'B01003_001E'),
        state_fips='*',
        county_fips='*',
        year=year)
    # Turn our received wa_census data into a pandas dataframe
    return pd.DataFrame(wa_census)


def tornado_census_by_year(years, tornadoes: pd.DataFrame):
    """
    Combines census data from retrieve_census_data for each year in years
    with the passed tornado information dataframe. Returns the resulting
    dataframe.
    """
    c = Census(API_KEY)
    result = None
    for year in years:
        # Filter the tornado down to only the same year as the census data
        t_filtered = pd.DataFrame(tornadoes[tornadoes['yr'] == year])
        # Get the cnesus data
        year_data = retrieve_census_data(c, year)
        # Drop Hawaii and Alaska data prior to joining
        year_data = year_data[(year_data['state'] != '02') &
                              (year_data['state'] != '15')]
        # Convert joined columns to the same dtype
        year_data['state'] = year_data['state'].apply(lambda x: int(x))
        year_data['county'] = year_data['county'].apply(lambda x: int(x))
        # Left join on both state and county to get a row for each tornado dp
        joined = t_filtered.merge(right=year_data, how='left',
                                  left_on=['stf', 'f1'],
                                  right_on=['state', 'county'])
        # Concatenate all data retrieved
        if result is None:
            result = joined
        else:
            result = pd.concat([result, joined], axis=0)
        result.to_csv('../data/joined.csv')
    return result


def import_tornado_data(path, hawaii=False, alaska=False, puerto_rico=False,
                        add_geometries=True, drop_dupes=True):
    """
    Reads the tornado csv dataset located at 'data/1950-2020_all_tornadoes.csv'
    into a dataframe, converts index to a datetime objet, drops duplicates,
    filters data by location, and creates start and end point geometries
    for geospatial data analysis of the tornadoes. Returns the resulting
    dataframe.
    """
    data = pd.read_csv(path, parse_dates=[['date', 'time']])
    data['stf'] = data['stf'].apply(lambda x: int(x))
    data['f1'] = data['f1'].apply(lambda x: int(x))
    data['yr'] = data['yr'].apply(lambda x: int(x))

    data.set_index(data['date_time'], inplace=True)
    if drop_dupes:
        data.drop_duplicates(subset=['date_time', 'st'], inplace=True)
    data = data[data['slon'] < 0]
    if not hawaii:
        data = data[data['stf'] != 15]
    if not alaska:
        data = data[data['stf'] != 2]
    if not puerto_rico:
        data = data[data['stf'] != 72]
    if add_geometries:
        add_start_end_points(data)
    return data


def import_state_geometries() -> gpd.GeoDataFrame:
    """
    Reads in the state json dataset located at 'data/gz_2010_us_040_00_5m.json'
    into a dataframe and filters out states that are far away from the lower 48
    U.S. states for easier geospatial analysis. Returns the resulting dataframe
    """
    states = gpd.read_file(STATE_FILE)
    states = states[(states['NAME'] != 'Alaska') &
                    (states['NAME'] != 'Hawaii') &
                    (states['NAME'] != 'Puerto Rico')]
    return states


def add_start_end_points(df):
    """
    Takes in a tornado information dataframe and adds start point and
    endpoint geometries. Modifies the dataframe in place.
    """
    df['start_point'] = [Point(lon, lat) for (lon, lat) in zip(df['slon'],
                                                               df['slat'])]
    df['end_point'] = [Point(lon, lat) for (lon, lat) in zip(df['elon'],
                                                             df['elat'])]


def main():
    pass


if __name__ == '__main__':
    main()

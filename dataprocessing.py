from census import Census
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


TORNADO_FILE = 'data/1950-2020_all_tornadoes.csv'
STATE_FILE = 'data/gz_2010_us_040_00_5m.json'
API_KEY = 'd3687e22fcd51ce480482a5caa07e0ae239c77a5'
ALL_VALID_YEARS = [x for x in range(2009, 2020)]


def retrieve_census_data(c: Census, year) -> pd.DataFrame:
    """
        FUNCTION COMMENT HERE
    """
    # Get a census object using the census library, api key provided for access
    # Extract Census data for specific categories on a per tract basis
        # C17002_001E: count of ratio of income to poverty in the past 12 months (total)
        # C17002_002E: count of ratio of income to poverty in the past 12 months (< 0.50)
        # C17002_003E: count of ratio of income to poverty in the past 12 months (0.50 - 0.99)
        # B01003_001E: total population
    #TRY TO QUERY FOR THE CORRECT CENSUS YEAR DATA year range (2009, 2019)
    wa_census = c.acs5.state_county(
        fields=('NAME', 'C17002_001E', 'C17002_002E', 'C17002_003E',
                'B01003_001E'),
        state_fips='*',
        county_fips='*',
        year=year)
    # Turn our received wa_census data into a pandas dataframe
    return pd.DataFrame(wa_census)

def tornado_census_by_year(years, tornadoes: pd.DataFrame) -> pd.DataFrame:
    c = Census(API_KEY)
    result = None
    for year in years:
        # Filter the tornado down to only the same year as the census data
        t_filtered = pd.DataFrame(tornadoes[tornadoes['yr'] == year])
        # Get the cnesus data
        year_data = retrieve_census_data(c, year)
        # Drop Hawaii and Alaska data prior to joining
        year_data = year_data[(year_data['state'] != '02') & (year_data['state'] != '15')]
        # Convert joined columns to the same dtype
        year_data['state'] = year_data['state'].apply(lambda x: int(x))
        year_data['county'] = year_data['county'].apply(lambda x: int(x))
        # Left join on both state and county to get a row for each tornado dp
        joined = t_filtered.merge(right=year_data, how='left', left_on=['stf', 'f1'], right_on=['state', 'county'])
        # Concatenate all data retrieved
        if result is None:
            result = joined
        else:
            pd.concat([result, joined], axis=0)
        print(year)
    return result

def import_tornado_data(hawaii=False, alaska=False, puerto_rico=False, add_geometries=True):
    data = pd.read_csv(TORNADO_FILE)
    data['stf'] = data['stf'].apply(lambda x: int(x))
    data['f1'] = data['f1'].apply(lambda x: int(x))
    data['yr'] = data['yr'].apply(lambda x: int(x))
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
    states = gpd.read_file(STATE_FILE)
    states = states[(states['NAME'] != 'Alaska') & (states['NAME'] != 'Hawaii') & (states['NAME'] != 'Puerto Rico')]
    return states

def add_start_end_points(df):
    # Add start point and end point geometries to joined pandas df using longitude and latitude
    df['start_point'] = [Point(lon, lat) for (lon, lat) in zip(df['slon'], df['slat'])]
    df['end_point'] = [Point(lon, lat) for (lon, lat) in zip(df['elon'], df['elat'])]

def get_processed_data():
    tornadoes = import_tornado_data()
    tornado_census = tornado_census_by_year([2017], tornadoes)

    fig, ax = plt.subplots(1, figsize=(20, 10))

    states = import_state_geometries()
    states.plot(ax=ax, color="#EEEEEE", edgecolor='black')

    geo_tornadoes = gpd.GeoDataFrame(data=tornadoes, geometry='start_point')
    geo_tornadoes.plot(ax=ax, column='mag', legend=True, markersize=tornadoes['mag'], vmin=0, vmax=5)
    plt.savefig('figures/testing.png')


def main():
    # Folium: https://www.analyticsvidhya.com/blog/2020/06/guide-geospatial-analysis-folium-python/
    # https://pygis.io/docs/d_access_census.html
    data = get_processed_data()

if __name__ == '__main__':
    main()
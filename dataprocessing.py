from census import Census
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

TORNADO_FILE = 'tornado-analysis/data/1950-2020_all_tornadoes.csv'
STATE_FILE = 'tornado-analysis/data/gz_2010_us_040_00_5m.json'
API_KEY = 'd3687e22fcd51ce480482a5caa07e0ae239c77a5'


def retrieve_census_data() -> pd.DataFrame:
    """
        FUNCTION COMMENT HERE
    """
    # Get a census object using the census library, api key provided for access
    c = Census(API_KEY)
    # Extract Census data for specific categories on a per tract basis
        # C17002_001E: count of ratio of income to poverty in the past 12 months (total)
        # C17002_002E: count of ratio of income to poverty in the past 12 months (< 0.50)
        # C17002_003E: count of ratio of income to poverty in the past 12 months (0.50 - 0.99)
        # B01003_001E: total population
    #TRY TO QUERY FOR THE CORRECT CENSUS YEAR DATA
    wa_census = c.acs5.state_county(
        fields=('NAME', 'C17002_001E', 'C17002_002E', 'C17002_003E',
                'B01003_001E'),
        state_fips=53,
        county_fips="*",
        year=2017)
    # Turn our received wa_census data into a pandas dataframe
    return pd.DataFrame(wa_census)


def import_tornado_data() -> pd.DataFrame:
    data = pd.read_csv(TORNADO_FILE)
    data = data[data['stf'] == 53] #filter down to Washington tornados
    return data


def import_state_geometries() -> gpd.GeoDataFrame:
    states = gpd.read_file(STATE_FILE)
    states = states[(states['NAME'] != 'Alaska') & (states['NAME'] != 'Hawaii')]
    return states


def get_processed_data():
    wa_census_data = retrieve_census_data()
    tornado_data = import_tornado_data()
    # convert columns to join on to the same type
    tornado_data['f1'] = tornado_data['f1'].apply(lambda x: int(x))
    wa_census_data['county'] = wa_census_data['county'].apply(lambda x: int(x))
    print(wa_census_data)
    print(tornado_data)
    joined = wa_census_data.merge(tornado_data, left_on='county', right_on='f1')
    # Add start point and end point geometries to joined pandas df using longitude and latitude
    joined['start_point'] = [Point(lon, lat) for (lon, lat) in zip(joined['slon'], joined['slat'])]
    joined['end_point'] = [Point(lon, lat) for (lon, lat) in zip(joined['elon'], joined['elat'])]
    print(joined['slon'].max())

    fig, ax = plt.subplots(1, figsize=(20, 10))

    states = import_state_geometries()
    states.plot(ax=ax, color="#EEEEEE", edgecolor='black', markersize=10)

    geo_plot_data = gpd.GeoDataFrame(data=joined, geometry='start_point')
    geo_plot_data.plot(ax=ax)
    plt.savefig('tornado-analysis/figures/testing.png')


def main():
    # Folium: https://www.analyticsvidhya.com/blog/2020/06/guide-geospatial-analysis-folium-python/
    # https://pygis.io/docs/d_access_census.html
    data = get_processed_data()



if __name__ == '__main__':
    main()
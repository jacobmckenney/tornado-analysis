from tokenize import group
import dataprocessing as dp
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

ALL_VALID_YEARS = [x for x in range(2009, 2020)]


def plot_magnitudes(tornadoes):
    tornado_census = dp.tornado_census_by_year(ALL_VALID_YEARS, tornadoes)
    dp.add_start_end_points(tornado_census)

    fig, ax = plt.subplots(1, figsize=(20, 10))
    plt.title('Tornado Location and Magnitude in the U.S (2009-2019)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    states = dp.import_state_geometries()
    states.plot(ax=ax, color="#EEEEEE", edgecolor='black')

    geo_tornadoes = gpd.GeoDataFrame(data=tornado_census, geometry='start_point')
    geo_tornadoes.plot(ax=ax, column='mag', legend=True, markersize=tornado_census['mag'], vmin=0, vmax=5)
    tornado_census.to_csv('data/joined.csv')
    plt.savefig('figures/2009-2019_magnitudes.png')

def most_in_year(tornadoes: pd.DataFrame):
    filtered = tornadoes[['yr', 'mo', 'dy']]
    grouped = filtered.groupby(by='yr')['yr'].count()
    max_year = grouped.idxmax()
    max_in_year = grouped.max()
    return (max_year, max_in_year)

def most_likely_day(tornadoes: pd.DataFrame):
    pass

def main(run_all):
    tornadoes = dp.import_tornado_data()
    print(tornadoes)
    print(most_in_year(tornadoes))
    if run_all:
        plot_magnitudes(tornadoes)

if __name__ == '__main__':
    main(True)
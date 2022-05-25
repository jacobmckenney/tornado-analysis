import dataprocessing as dp
import matplotlib.pyplot as plt
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

def main():
    tornadoes = dp.import_tornado_data()
    plot_magnitudes(tornadoes)

if __name__ == '__main__':
    main()
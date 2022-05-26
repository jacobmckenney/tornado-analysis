import dataprocessing as dp
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


ALL_VALID_YEARS = [x for x in range(2009, 2020)]


def plot_magnitudes(tornadoes, states):
    tornado_census = dp.tornado_census_by_year(ALL_VALID_YEARS, tornadoes)
    dp.add_start_end_points(tornado_census)

    fig, ax = plt.subplots(1, figsize=(20, 10))
    plt.title('Tornado Location and Magnitude in the U.S (2009-2019)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    states.plot(ax=ax, color="#EEEEEE", edgecolor='black')

    geo_tornadoes = gpd.GeoDataFrame(data=tornado_census, geometry='start_point')
    geo_tornadoes.plot(ax=ax, column='mag', legend=True, markersize=tornado_census['mag'], vmin=0, vmax=5)
    tornado_census.to_csv('data/joined.csv')
    plt.savefig('figures/magnitudes_2009-2019.png')

def plot_tornadoes_by_state(tornadoes: gpd.GeoDataFrame, states: pd.DataFrame):
    fig, [ax1, ax2] = plt.subplots(2, figsize=(10, 10))
    ax1.set_title('Tornados per State in the U.S. (1950-2020)')
    ax2.set_title('Tornados per Sq Mile by State in the U.S. (1950-2020)') # CHECK
    tornadoes['count'] = tornadoes['yr']
    tornadoes = tornadoes[['stf', 'count']]
    tornado_state_counts = tornadoes.groupby(by='stf').count()
    states['STATE'] = states['STATE'].apply(lambda x: int(x))
    # Merge counts of tornadoes per state into state data to be plotted
    merged = states.merge(right=tornado_state_counts, how='left', left_on='STATE', right_on='stf')
    merged['normalized_count'] = merged['count'] / merged['CENSUSAREA']
    merged.plot(ax=ax1, column='count', legend=True)
    merged.plot(ax=ax2, column='normalized_count', legend=True)
    plt.savefig('figures/tornado_count_1950-2020.png')



def most_in_year(tornadoes: pd.DataFrame):
    filtered = tornadoes[['yr', 'mo', 'dy']]
    grouped = filtered.groupby(by='yr')['yr'].count()
    max_year = grouped.idxmax()
    max_in_year = grouped.max()
    return (max_year, max_in_year)

def most_likely_time_period(index, timeperiod, figname, df: pd.DataFrame):
    tornadoes = pd.DataFrame(df)
    tornadoes['count'] = tornadoes['yr']
    tornadoes = tornadoes[['count']]
    # Use groupby because we want to plot year-agnostic data
    time_tornadoes = tornadoes.groupby(index).count()
    sns.relplot(data=time_tornadoes, x='date_time', y='count', kind='line')
    plt.xlabel(timeperiod)
    plt.savefig(f'figures/{figname}.png', bbox_inches='tight')

def devestation_predictions(df):
    tornadoes = pd.DataFrame(df)
    labels = tornadoes[[]]
    pass



def main(run_all):
    tornadoes = dp.import_tornado_data()
    states = dp.import_state_geometries()
    most_likely_time_period(tornadoes.index.month, 'month', 'monthly', tornadoes)
    most_likely_time_period(tornadoes.index.week, 'week', 'weekly', tornadoes)
    most_likely_time_period(tornadoes.index.dayofyear, 'day', 'daily', tornadoes)
    plot_tornadoes_by_state(tornadoes, states)
    if run_all:
        print(most_in_year(tornadoes))
        plot_magnitudes(tornadoes, states)

if __name__ == '__main__':
    main(False)
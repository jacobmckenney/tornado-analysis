"""
Jacob McKenney & Luke Sala

Performs analysis on dataframes that can be retrieved from the methods
located in dataprocessing.py. Analysis incldes: normalized tornado count
by state, finding most tornadoes in a year, training and tuning machine
learning models to predict devastation factors, plotting tornadoes by
magnitude and understanding demographics of high tornado-density areas.

NOTE: When a parameter is described as a dataframe containing tornado
information (or tornado dataframe) this dataframe should have been obtained via
dataprocessing.import_tornado_data() to ensure proper functionality
"""
import dataprocessing as dp
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import tuning_args as ta


ALL_VALID_YEARS = [x for x in range(2009, 2020)]
DEVASTATION_FEATURES = ['mo', 'dy', 'mag', 'stf', 'loss', 'closs', 'slat',
                        'slon', 'len', 'wid', 'fat']
TORNADO_FILE = 'data/1950-2020_all_tornadoes.csv'
RESULT_FILE = 'data/analysis-results.txt'
MAGNITUDES_PATH = 'figures/magnitudes_2009-2019.png'


def plot_tornadoes_by_state(tornadoes, states):
    """
    Takes in a geodataframe containing information about tornadoes
    and a geodataframestates containing information about U.S. state
    geometries and plots two graphs in one file
    'figures/tornado_count_1950-2020.png' with one
    graphing showing the raw count of tornadoes per state from 1950 to 2020
    and the other graph showing the normalized count (count / square miles)
    for each state from 1950 to 2020
    """
    fig, [ax1, ax2] = plt.subplots(2, figsize=(10, 10))
    ax1.set_title('Tornadoes by State in the U.S. (1950-2020)')
    ax2.set_title('Tornadoes per Sq Mile by State in the U.S. (1950-2020)')
    tornadoes['count'] = tornadoes['yr']
    tornadoes = tornadoes[['stf', 'count']]
    tornado_state_counts = tornadoes.groupby(by='stf').count()
    states['STATE'] = states['STATE'].apply(lambda x: int(x))
    # Merge counts of tornadoes per state into state data to be plotted
    merged = states.merge(right=tornado_state_counts, how='left',
                          left_on='STATE', right_on='stf')
    merged['normalized_count'] = merged['count'] / merged['CENSUSAREA']
    merged.plot(ax=ax1, column='count', legend=True)
    merged.plot(ax=ax2, column='normalized_count', legend=True)
    plt.savefig('figures/tornado_count_1950-2020.png')


def most_in_year(tornadoes):
    """
    Takes in a tornadoes dataframe and finds the year with the most
    tornado occurences across the United States and the number of
    tornadoes occuring that year. Returns this information as a tuple:
    (number, number) -> (year, year_count)
    """
    filtered = tornadoes[['yr', 'mo', 'dy']]
    grouped = filtered.groupby(by='yr')['yr'].count()
    max_year = grouped.idxmax()
    max_in_year = grouped.max()
    return (max_year, max_in_year)


def most_likely_time_period(index, timeperiod, figname, df: pd.DataFrame):
    """
    Given an datetime index of the passed dataframe(df) which should be a
    tornado dataframe and information about that index (timeperiod - str and
    figname -str) plots the information about tornadoes within that datetime
    index (i.e weekly, monthly, etc.).
    """
    tornadoes = pd.DataFrame(df)
    tornadoes['count'] = tornadoes['yr']
    tornadoes = tornadoes[['count']]
    # Use groupby because we want to plot year-agnostic data
    time_tornadoes = tornadoes.groupby(index).count()
    sns.relplot(data=time_tornadoes, x='date_time', y='count', kind='line')
    plt.xlabel(timeperiod)
    plt.savefig(f'figures/{figname}.png', bbox_inches='tight')


def devastation_predictions(df, quick_tune=False):
    """
    Takes in a tornado dataframe, df, and a boolean, quick_tune, and attempts
    to predict various devastation metrics about a tornado based on the
    tornado dataset. Uses and trains multiple kinds of machine learning models
    to see which one can best predict these metrics. Writes model accuracy
    information to the file 'data/analysis-results.txt' using mean squared
    error as the accuracy metric. Models used: DecisionTreeRegressor,
    KNeighborsRegressor, and LinearRegression
    """
    tornadoes = pd.DataFrame(df)
    tornadoes.index = range(len(tornadoes.index))
    tornadoes[DEVASTATION_FEATURES] = \
        tornadoes[DEVASTATION_FEATURES].astype(float)
    features = tornadoes[DEVASTATION_FEATURES]
    tornadoes[['inj']] = tornadoes[['inj']].astype(float)
    labels = tornadoes[['inj']]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    fit_kwargs = {
        'features_train': features_train,
        'features_test': features_test,
        'labels_train': labels_train
    }

    t_tp = ta.TREE_TUNING_QUICK if quick_tune else ta.TREE_TUNING_FULL
    kn_tp = ta.KNEIGHBORS_TUNING_QUICK if quick_tune \
        else ta.KNEIGHBORS_TUNING_FULL

    results = []
    decision_not_tuned = fit_and_test_model(DecisionTreeRegressor(),
                                            **fit_kwargs)
    results.append((decision_not_tuned, False, 'DecisionTree'))

    gsv = GridSearchCV(estimator=DecisionTreeRegressor(),
                       param_grid=t_tp,
                       scoring='neg_mean_squared_error', cv=3, verbose=2)
    gsv.fit(features_train, labels_train)
    decision_tuned = fit_and_test_model(
        DecisionTreeRegressor(**gsv.best_params_), **fit_kwargs)
    results.append((decision_tuned, True, 'DecisionTree'))

    kneighbors_not_tuned = fit_and_test_model(
        KNeighborsRegressor(), **fit_kwargs)
    results.append((kneighbors_not_tuned, False, 'KNeighbors'))

    kn_gsv = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid=kn_tp,
                          scoring='neg_mean_squared_error', cv=3, verbose=2)
    kn_gsv.fit(features_train, labels_train)
    kneighbors_tuned = fit_and_test_model(
        KNeighborsRegressor(**kn_gsv.best_params_), **fit_kwargs)
    results.append((kneighbors_tuned, True, 'KNeighbors'))

    lr_not_tuned = fit_and_test_model(LinearRegression(), **fit_kwargs)
    results.append((lr_not_tuned, False, 'LinearRegression'))

    o_sys = sys.stdout
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        for result in results:
            print(f'{result[2]} - Train Accuracy,',
                  ('Tuned:' if result[1] else 'Not-Tuned:'),
                  mean_squared_error(result[0][0], labels_train))
            print(f'{result[2]} - Test Accuracy,',
                  ('Tuned:' if result[1] else 'Not-Tuned:'),
                  mean_squared_error(result[0][1], labels_test), '\n')
        print('\n')
        sys.stdout = o_sys


def fit_and_test_model(model, features_train, features_test, labels_train):
    """
    Takes in a model instance (already instantiated with hyperparameters) and
    trains the model on the provided training set and then returns the
    predictions for the training set and the test set.
    """
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)
    return (train_predictions, test_predictions)


def plot_magnitudes(df, states, save_path, years=ALL_VALID_YEARS):
    """
    Takes in dataframes containing tornado information and state geometries
    and plots a graph of the magnitudes of tornadoes over the years specified
    by the years list (default 2009-2019).
    """
    tornadoes = gpd.GeoDataFrame(df)
    tornadoes = tornadoes.query('yr in @years')
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plt.title('Tornado Location and Magnitude in the U.S (2009-2019)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    states.plot(ax=ax, color='#EEEEEE', edgecolor='black')
    geo_tornadoes = gpd.GeoDataFrame(data=tornadoes, geometry='start_point')
    geo_tornadoes.plot(ax=ax, column='mag', legend=True,
                       markersize=(tornadoes['mag'] * 3), vmin=0, vmax=5)
    plt.savefig(save_path)


def poverty_and_tornadoes(census):
    """
    Takes in a tornado information dataframe combined with census data to
    try to find trends in demographic information in tornado-heavy counties.
    The goal is to learn more about the areas that tornadoes most affect.
    Results are written to the 'data/analysis-results.txt' file.
    """
    unique_counties = census.drop_duplicates(subset=['NAME'])
    county_counts = census.groupby('NAME').count()
    top_10_percent = county_counts.nlargest((round(len(county_counts) * 0.1)),
                                            'county')
    top_10_percent['count'] = top_10_percent['yr']
    top_10_percent.reset_index(inplace=True, drop=False)
    top_10_percent = top_10_percent[['NAME', 'count']]
    bottom_90_percent = \
        county_counts.nsmallest((round(len(county_counts) * 0.9)), 'county')
    bottom_90_percent['count'] = bottom_90_percent['yr']
    bottom_90_percent.reset_index(inplace=True, drop=False)
    bottom_90_percent = bottom_90_percent[['NAME', 'count']]
    top_10_merged = top_10_percent.merge(right=unique_counties, left_on='NAME',
                                         right_on='NAME', how='left')
    bottom_90_merged = bottom_90_percent.merge(right=unique_counties,
                                               left_on='NAME', right_on='NAME',
                                               how='left')
    o_sys = sys.stdout
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        print('Average median income of top 10 percent:',
              top_10_merged['B19013_001E'].mean())
        print('Average median income of bottom 90 percent:',
              bottom_90_merged['B19013_001E'].mean(), '\n')
        sys.stdout = o_sys


def main(args):
    """
    Uses methods from analysis.py and dataprocessing.py to
    perform a complete analysis on tornadoes and extract pertinent
    information/graphs.
    """
    tornadoes = dp.import_tornado_data(TORNADO_FILE)
    states = dp.import_state_geometries()
    census = dp.tornado_census_by_year(ALL_VALID_YEARS, tornadoes)
    o_sys = sys.stdout
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        print('Most tornadoes in a year (year, count):',
              most_in_year(tornadoes), '\n')
    sys.stdout = o_sys
    poverty_and_tornadoes(census)
    plot_magnitudes(tornadoes, states, MAGNITUDES_PATH)
    devastation_predictions(tornadoes,
                            quick_tune=(not args[0] if len(args) == 1
                                        else True))
    tornadoes = dp.import_tornado_data(TORNADO_FILE)
    most_likely_time_period(tornadoes.index.year, 'year',
                            'yearly', tornadoes)
    most_likely_time_period(tornadoes.index.month, 'month',
                            'monthly', tornadoes)
    most_likely_time_period(tornadoes.index.week, 'week',
                            'weekly', tornadoes)
    most_likely_time_period(tornadoes.index.dayofyear, 'day',
                            'daily', tornadoes)
    plot_tornadoes_by_state(tornadoes, states)


if __name__ == '__main__':
    main(sys.argv[1:])

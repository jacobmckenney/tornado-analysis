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
                        'slon', 'len', 'wid', 'fat', 'inj']
TORNADO_FILE = '../data/1950-2020_all_tornadoes.csv'
RESULT_FILE = '../data/analysis-results.txt'
MAGNITUDES_PATH = '../figures/magnitudes_2009-2019.png'
COUNT_PATH = '../figures/tornado_count_1950-2020.png'


def plot_tornadoes_by_state(tornadoes, states, save_path):
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
    states.plot(ax=ax1, color='#EEEEEE', edgecolor='black')
    states.plot(ax=ax2, color='#EEEEEE', edgecolor='black')
    merged = states.merge(right=tornado_state_counts, how='left',
                          left_on='STATE', right_on='stf')
    merged['normalized_count'] = merged['count'] / merged['CENSUSAREA']
    merged.plot(ax=ax1, column='count', legend=True)
    merged.plot(ax=ax2, column='normalized_count', legend=True)
    plt.savefig(save_path)


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


def most_likely_time_period(index, timeperiod, save_path, df: pd.DataFrame):
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
    plt.savefig(save_path, bbox_inches='tight')


def devastation_predictions(df, label, quick_tune=False):
    """
    Takes in a tornado dataframe, df, a label to predict, label,
    and a boolean, quick_tune, and attempts
    to predict various devastation metrics about a tornado based on the
    tornado dataset. Uses and trains multiple kinds of machine learning models
    to see which one can best predict these metrics. Writes model accuracy
    information to the file 'data/analysis-results.txt' using mean squared
    error as the accuracy metric. Models used: DecisionTreeRegressor,
    KNeighborsRegressor, and LinearRegression
    """
    tornadoes = pd.DataFrame(df)
    tornadoes.index = range(len(tornadoes.index))
    feature_cols = [x for x in DEVASTATION_FEATURES if x != label]
    tornadoes[DEVASTATION_FEATURES] = \
        tornadoes[DEVASTATION_FEATURES].astype(float)
    features = tornadoes[feature_cols]
    labels = tornadoes[[label]]
    print(features.columns)
    print(labels.columns)
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
    results.append((decision_tuned, True, 'DecisionTree', gsv.best_params_))

    kneighbors_not_tuned = fit_and_test_model(
        KNeighborsRegressor(), **fit_kwargs)
    results.append((kneighbors_not_tuned, False, 'KNeighbors'))

    kn_gsv = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid=kn_tp,
                          scoring='neg_mean_squared_error', cv=3, verbose=2)
    kn_gsv.fit(features_train, labels_train)
    kneighbors_tuned = fit_and_test_model(
        KNeighborsRegressor(**kn_gsv.best_params_), **fit_kwargs)
    results.append((kneighbors_tuned, True, 'KNeighbors', kn_gsv.best_params_))

    lr_not_tuned = fit_and_test_model(LinearRegression(), **fit_kwargs)
    results.append((lr_not_tuned, False, 'LinearRegression'))

    o_sys = sys.stdout
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        print('Predicting:', label, '\n')
        for result in results:
            if result[1]:
                print('Best parameters:', result[3])
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


def poverty_and_tornadoes(census, column):
    """
    Takes in a tornado information dataframe combined with census data to
    try to find trends in demographic information in tornado-heavy counties.
    The goal is to learn more about the areas that tornadoes most affect.
    Results are written to the 'data/analysis-results.txt' file.
    """
    unique_counties = census.drop_duplicates(subset=['NAME'])
    county_counts = census.groupby('NAME').count()
    for x in range(10, 51, 10):
        top = county_counts.nlargest((round(len(county_counts) * (x / 100))),
                                     'county')
        top['count'] = top['county']
        top.reset_index(inplace=True, drop=False)
        top = top[['NAME', 'count']]
        bottom = county_counts.nsmallest((round(len(county_counts) *
                                         (1 - x / 100))), 'county')
        bottom['count'] = bottom['yr']
        bottom.reset_index(inplace=True, drop=False)
        bottom = bottom[['NAME', 'count']]
        top_merged = top.merge(right=unique_counties, left_on='NAME',
                               right_on='NAME', how='left')
        bottom_merged = bottom.merge(right=unique_counties, left_on='NAME',
                                     right_on='NAME', how='left')
        o_sys = sys.stdout
        with open(RESULT_FILE, 'a') as f:
            sys.stdout = f
            print(f'Top {x} percent of counties (tornado-density):',
                  top_merged[column].mean())
            print(f'Bottom {100 - x} percent of counties (tornado-density):',
                  bottom_merged[column].mean(), '\n')
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
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        print('Metric: average median income:')
    sys.stdout = o_sys
    poverty_and_tornadoes(census, 'B19013_001E')
    with open(RESULT_FILE, 'a') as f:
        sys.stdout = f
        print('Metric: ratio of income to poverty in last 12 months:')
    sys.stdout = o_sys
    poverty_and_tornadoes(census, 'C17002_001E')
    plot_magnitudes(tornadoes, states, MAGNITUDES_PATH)
    quick_tune = (not args[0] if len(args) == 1 else True)
    devastation_predictions(tornadoes, 'inj', quick_tune)
    devastation_predictions(tornadoes, 'fat', quick_tune)
    devastation_predictions(tornadoes, 'loss', quick_tune)
    tornadoes = dp.import_tornado_data(TORNADO_FILE)
    most_likely_time_period(tornadoes.index.year, 'year',
                            '../figures/yearly.png', tornadoes)
    most_likely_time_period(tornadoes.index.month, 'month',
                            '../figures/monthly.png', tornadoes)
    most_likely_time_period(tornadoes.index.week, 'week',
                            '../figures/weekly.png', tornadoes)
    most_likely_time_period(tornadoes.index.dayofyear, 'day',
                            '../figures/daily.png', tornadoes)
    plot_tornadoes_by_state(tornadoes, states, COUNT_PATH)


if __name__ == '__main__':
    main(sys.argv[1:])

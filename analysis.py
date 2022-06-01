import dataprocessing as dp
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import sys
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


ALL_VALID_YEARS = [x for x in range(2009, 2020)]
DEVASTATION_FEATURES = ['mo', 'dy', 'mag', 'stf', 'loss', 'closs', 'slat', 'slon', 'len', 'wid', 'fat']

def plot_tornadoes_by_state(tornadoes: gpd.GeoDataFrame, states: pd.DataFrame):
    fig, [ax1, ax2] = plt.subplots(2, figsize=(10, 10))
    ax1.set_title('Tornadoes by State in the U.S. (1950-2020)')
    ax2.set_title('Tornadoes per Sq Mile by State in the U.S. (1950-2020)')
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

def devastation_predictions(df, quick_tune=False):
    tornadoes = pd.DataFrame(df)
    tornadoes.index = range(len(tornadoes.index))
    tornadoes[DEVASTATION_FEATURES] = tornadoes[DEVASTATION_FEATURES].astype(float)
    features = tornadoes[DEVASTATION_FEATURES]
    tornadoes[['inj']] = tornadoes[['inj']].astype(float)
    labels = tornadoes[['inj']]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    tree_tuning_params = {
        'splitter':['best'],
        'max_depth' : [1, 3],
        'min_samples_leaf':[9, 10],
        'min_weight_fraction_leaf':[0.1, 0.2],
        'max_features':['auto'],
        'max_leaf_nodes':[10, 20]
    } if quick_tune else {
        'splitter':['best','random'],
        'max_depth' : [1, 3, 5, 7, 9, 11, 12],
        'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_weight_fraction_leaf':[0.1, 0.2, 0.3, 0.4, 0.5],
        'max_features':['auto','log2','sqrt'],
        'max_leaf_nodes':[10, 20, 30, 40, 50, 60, 70, 80, 90]
    }
    fit_kwargs = {'features_train': features_train, 'features_test': features_test, 'labels_train': labels_train}

    kneighbors_tuning_params = {'n_neighbors': [2,3,4], 'weights': ['uniform', 'distance'], 'p': [1, 2]} if quick_tune else {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'weights': ['uniform','distance'],'p':[1,2,5]}
    results = []
    decision_not_tuned = fit_and_test_model(DecisionTreeRegressor(), **fit_kwargs)
    results.append((decision_not_tuned, False, 'DecisionTree'))

    gsv = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=tree_tuning_params, scoring='neg_mean_squared_error', cv=3, verbose=2)
    gsv.fit(features_train, labels_train)
    decision_tuned = fit_and_test_model(DecisionTreeRegressor(**gsv.best_params_), **fit_kwargs)
    results.append((decision_tuned, True, 'DecisionTree'))

    kneighbors_not_tuned = fit_and_test_model(KNeighborsRegressor(), **fit_kwargs)
    results.append((kneighbors_not_tuned, False, 'KNeighbors'))

    kn_gsv = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=kneighbors_tuning_params, scoring='neg_mean_squared_error', cv=3, verbose=2)
    kn_gsv.fit(features_train, labels_train)
    kneighbors_tuned = fit_and_test_model(KNeighborsRegressor(**kn_gsv.best_params_), **fit_kwargs)
    results.append((kneighbors_tuned, True, 'KNeighbors'))

    lr_not_tuned = fit_and_test_model(LinearRegression(), **fit_kwargs)
    results.append((lr_not_tuned, False, 'LinearRegression'))

    for result in results:
        print(f'{result[2]} - Train Accuracy,', ('Tuned:' if result[1] else 'Not-Tuned:'), mean_squared_error(result[0][0], labels_train))
        print(f'{result[2]} - Test Accuracy,', ('Tuned:' if result[1] else 'Not-Tuned:'), mean_squared_error(result[0][1], labels_test), '\n')

def fit_and_test_model(model, features_train, features_test, labels_train):
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)
    return (train_predictions, test_predictions)

def plot_magnitudes(df: gpd.GeoDataFrame, states, years=ALL_VALID_YEARS):
    tornadoes = gpd.GeoDataFrame(df)
    tornadoes = tornadoes.query('yr in @years')
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plt.title('Tornado Location and Magnitude in the U.S (2009-2019)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    states.plot(ax=ax, color='#EEEEEE', edgecolor='black')

    geo_tornadoes = gpd.GeoDataFrame(data=tornadoes, geometry='start_point')
    geo_tornadoes.plot(ax=ax, column='mag', legend=True, markersize=tornadoes['mag'], vmin=0, vmax=5)
    plt.savefig('figures/magnitudes_2009-2019.png')

def poverty_and_tornadoes(census: pd.DataFrame, states: gpd.GeoDataFrame):
    unique_counties = census.drop_duplicates(subset=['NAME'])
    county_counts = census.groupby('NAME').count()
    top_10_percent = county_counts.nlargest((round(len(county_counts) * 0.1)), 'county')
    top_10_percent['count'] = top_10_percent['yr']
    top_10_percent.reset_index(inplace=True, drop=False)
    top_10_percent = top_10_percent[['NAME', 'count']]
    bottom_90_percent = county_counts.nsmallest((round(len(county_counts) * 0.9)), 'county')
    bottom_90_percent['count'] = bottom_90_percent['yr']
    bottom_90_percent.reset_index(inplace=True, drop=False)
    bottom_90_percent = bottom_90_percent[['NAME', 'count']]
    top_10_merged = top_10_percent.merge(right=unique_counties, left_on='NAME', right_on='NAME', how='left')
    bottom_90_merged = bottom_90_percent.merge(right=unique_counties, left_on='NAME', right_on='NAME', how='left')
    print('Average median income of top 10 percent:', top_10_merged['B19013_001E'].mean())
    print('Average median income of bottom 90 percent:', bottom_90_merged['B19013_001E'].mean())


def main(args):
    tornadoes = dp.import_tornado_data()
    states = dp.import_state_geometries()
    census = dp.tornado_census_by_year(ALL_VALID_YEARS, tornadoes)
    poverty_and_tornadoes(census, states)
    plot_magnitudes(tornadoes, states)
    if args[0]:
        devastation_predictions(tornadoes, quick_tune=(args[1] if args[1] else True))
        tornadoes = dp.import_tornado_data()
        most_likely_time_period(tornadoes.index.month, 'month', 'monthly', tornadoes)
        most_likely_time_period(tornadoes.index.week, 'week', 'weekly', tornadoes)
        most_likely_time_period(tornadoes.index.dayofyear, 'day', 'daily', tornadoes)
        plot_tornadoes_by_state(tornadoes, states)
        print(most_in_year(tornadoes))

if __name__ == '__main__':
    main(sys.argv[1:])
"""
Jacob McKenney & Luke Sala

Defines methods for testing the validity of our data analysis functioins
in analysis.py
"""
import dataprocessing as dp
import analysis as a
from cse163_utils import assert_equals


TESTING_TORNADO_FULL = 'test_data/tornado_test_all.csv'
MAGNITUDES_PATH = 'test_figures/magnitudes_2009-2019.png'
COUNT_PATH = 'test_figures/tornado_count_1950-2020.png'


def main():
    full_tornadoes = dp.import_tornado_data(TESTING_TORNADO_FULL)
    states = dp.import_state_geometries()
    # Test file includes 3 tornadoes from 2009-2019 result should plot
    # one tornado in alabama, arkansas, and florida
    a.plot_magnitudes(full_tornadoes, states, MAGNITUDES_PATH)
    # Testing file includes 3 rows for Texas but should only plot a value
    # of 2 tornadoes for TX because two are duplicates (different legs
    # of the same tornado)
    a.plot_tornadoes_by_state(full_tornadoes, states, COUNT_PATH)
    # There are four rows for 1995 but again one is a duplicate
    assert_equals((1995, 3), a.most_in_year(full_tornadoes))
    # Check to make sure these graphs make sense for the small dataset
    a.most_likely_time_period(full_tornadoes.index.year, 'year',
                              'test_figures/yearly.png', full_tornadoes)
    a.most_likely_time_period(full_tornadoes.index.month, 'month',
                              'test_figures/monthly.png', full_tornadoes)
    a.most_likely_time_period(full_tornadoes.index.week, 'week',
                              'test_figures/weekly.png', full_tornadoes)
    # Dataset includes two tornadoes on same day different year which should
    # cause a spike
    a.most_likely_time_period(full_tornadoes.index.dayofyear, 'day',
                              'test_figures/daily.png', full_tornadoes)


if __name__ == '__main__':
    main()

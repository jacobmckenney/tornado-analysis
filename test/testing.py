"""
Jacob McKenney & Luke Sala

Defines methods for testing the validity of our data analysis functioins
in analysis.py
"""
import dataprocessing as dp
import analysis as a


TESTING_TORNADO_FULL = 'tornado_test_all.csv'
MAGNITUDES_PATH = 'test_figures/magnitudes_2009-2019.png'


def main():
    full_tornadoes = dp.import_tornado_data(TESTING_TORNADO_FULL)
    states = dp.import_state_geometries()
    a.plot_magnitudes(full_tornadoes, states, MAGNITUDES_PATH)


if __name__ == '__main__':
    main()

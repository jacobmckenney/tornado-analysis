# Tornado Analysis

## Project Members
- Jacob McKenney (1778020)
- Luke Sala (1838050)

## Description
### Files & Folders
- `dataprocessing.py`: Imports, retrieves, cleans, filters, and joins datasets
- `analysis.py`: Runs analysis on dataframes retrieved from `dataprocessing.py`
- `tuning_args.py`: Stores static tuning parameters for different tuning
variations
- `data`: Folder containing all data that is imported or written during
analysis. Also includes a pdf explaining the tornado dataset used
- `figures`: Folder containing figures that are produced by running
`analysis.py`

## Running Our Script
### Required Packages
Make sure to have these packages installed in the virtual environment you
are running `analysis.py`
- pandas
- geopandas
- seaborn
- matplotlib.pyplot
- sys
- os
- sklearn
- Census
- shapely.geometry

Our script optionally takes in an argument via the command line which
controls whether or not to use the full tuning parameters for tuning the
machine learning models. If you run the script with no arguments it
will use a less comprehensive set of tuning parameters to improve runtime. To
run the full tuning parameters use the following command:
`python analysis.py True`
Where `python` is the path to the proper python executable for your env.
Else just run:
`python analysis.py`
An example run command for our env is: `/Users/jacobmckenney/opt/anaconda3/envs/cse163/bin/python /Users/jacobmckenney/Desktop/cs/cse163/project/tornado_analysis/src/analysis.py True`

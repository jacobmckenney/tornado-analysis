# Tornado Analysis

## Project Members
- Jacob McKenney (1778020)
- Luke Sala (1838050)

## Description
This project uses data found on https://www.spc.noaa.gov/wcm/#data about
tornadoes from 1950-2020 to do analysis and answer questions and predict
tornado trends.
### Files & Folders
- `dataprocessing.py`: Imports, retrieves, cleans, filters, and joins datasets
- `analysis.py`: Runs analysis on dataframes retrieved from `dataprocessing.py`
- `tuning_args.py`: Stores static tuning parameters for different tuning
variations
- `data`: Folder containing all data that is imported or written during
analysis. Also includes a pdf explaining the tornado dataset used
- `figures`: Folder containing figures that are produced by running
`analysis.py`
- `testing.py`: Script that tests our functions against a smaller dataset
- `test_data`: Directory holding test files
- `test_figures`: Figures created by our testing script
- `final_report.pdf`: Our report for our analysis

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
- Census (install with `conda install -c conda-forge census`)
- shapely.geometry

### Command Line
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
IMPORTANT: make sure you are in the src directory under tornado-analysis when running the script

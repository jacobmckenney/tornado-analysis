# Tornado Analysis

## Project Members
- Jacob McKenney (1778020)
- Luke Sala (1838050)

## Description

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
<center>`python analysis.py True`</center>
Where `python` is the path to the proper python executable for your env.
Else just run:
<center>`python analysis.py</center>
from census import Census
import pandas as pd

def main():
    # Folium: https://www.analyticsvidhya.com/blog/2020/06/guide-geospatial-analysis-folium-python/
    # https://pygis.io/docs/d_access_census.html
    c = Census("d3687e22fcd51ce480482a5caa07e0ae239c77a5")
    wa_census = c.acs5.state_county_tract(
        fields=('NAME', 'C17002_001E', 'C17002_002E', 'C17002_003E',
                'B01003_001E'),
        state_fips=53,
        county_fips="*",
        tract="*",
        year=2017)
    df = pd.DataFrame(wa_census)
    print(df.head())

if __name__ == '__main__':
    main()
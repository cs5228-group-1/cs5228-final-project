import pandas as pd


def preprocess_v1(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the following attributes:
    - rent_approval_date
    - block
    - street name
    - furnished
    - elevation
    - planning_area
    - region
    - latitude
    - longitude
    """
    dataframe = dataframe\
        .drop(columns=['rent_approval_date', 'block', 'street_name', 'furnished', 'elevation', 'planning_area', 'region', 'latitude', 'longitude'])\
        .replace({'flat_type': {'-': ' '}})

    return dataframe



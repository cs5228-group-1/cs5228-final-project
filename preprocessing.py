import pandas as pd
import numpy as np
from numpy import arccos, cos, sin
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

MRT_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-mrt-existing-stations.csv"
SHOPPING_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-shopping-malls.csv"
POSITION_ATTRS = ['latitude', 'longitude']


def preprocess_v1(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the following attributes:
    - rent_approval_date
    - town
    - block
    - street name
    - furnished
    - elevation
    - planning_area
    - region
    """
    reg_dates = pd.to_datetime(dataframe['rent_approval_date'], format="%Y-%m")
    dataframe = dataframe\
        .assign(
            year=reg_dates.dt.year - reg_dates.dt.year.min(),
            month=reg_dates.dt.month,
            flat_type=lambda df: df.flat_type.str.replace('-', ' '),
        )\
        .drop(columns=[
            'rent_approval_date',
            'town',
            'block',
            'street_name',
            'furnished',
            'elevation',
            'planning_area',
            'region',
        ])

    return dataframe


def great_circle_distance(
        lat1: float, lng1: float,
        lat2: float, lng2: float) -> float:
    R = 3963.0
    lat1, lng1 = np.deg2rad(lat1), np.deg2rad(lng1)
    lat2, lng2 = np.deg2rad(lat2), np.deg2rad(lng2)
    return R * arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))


def distance_to_nearest_place(row, df, code_col):
    """
    Return:
    - Distance (in km)
    - Code/Name
    """
    lat, lng = row.latitude, row.longitude

    distances = great_circle_distance(
        lat, lng, df.latitude.values, df.longitude.values)

    min_distance = np.nanmin(distances) * 1000
    min_idx = np.nanargmin(distances)

    return min_distance, df[code_col].iloc[min_idx]


def preprocess_v2(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Same as v1 with additional distance from the apartment
    to the closest existing MRTs.

    Note: The current implementation is not optimal.
    """
    mrt_df = pd.read_csv(MRT_DATAFRAME_PATH)
    dataframe = preprocess_v1(dataframe)
    dataframe[['nearest_mrt_dist', 'nearest_mrt_code']] = \
        dataframe.apply(
            lambda row: distance_to_nearest_place(row, mrt_df, 'code'),
            axis=1,
            result_type="expand"
    )
    return dataframe


def preprocess_v3(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Same as v2 with additional distance from the apartment
    to the closest existing shopping malls.

    Note: The current implementation is not optimal.
    """
    dataframe = preprocess_v1(dataframe)

    mrt_df = pd.read_csv(MRT_DATAFRAME_PATH).drop_duplicates(POSITION_ATTRS)
    dataframe[['nearest_mrt_dist', 'nearest_mrt_code']] = \
        dataframe.parallel_apply(
            lambda row: distance_to_nearest_place(row, mrt_df, 'code'),
            axis=1,
            result_type="expand"
    )

    mall_df = pd.read_csv(SHOPPING_DATAFRAME_PATH).drop_duplicates(POSITION_ATTRS)
    dataframe[['nearest_mall_dist', 'nearest_mall_name']] = \
        dataframe.parallel_apply(
            lambda row: distance_to_nearest_place(row, mall_df, 'name'),
            axis=1,
            result_type="expand"
    )
    return dataframe

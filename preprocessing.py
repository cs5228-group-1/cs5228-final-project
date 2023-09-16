import pandas as pd
import numpy as np
from numpy import arccos, cos, sin
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

MRT_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-mrt-existing-stations.csv"


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
    reg_dates = pd.to_datetime(dataframe['rent_approval_date'], format="%Y-%m")
    dataframe = dataframe\
        .assign(
            year=reg_dates.dt.year,
            month=reg_dates.dt.month,
            flat_type=lambda df: df.flat_type.str.replace('-', ' ')
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
            # 'latitude',
            # 'longitude'
        ])

    return dataframe


def great_circle_distance(
        lat1: float, lng1: float,
        lat2: float, lng2: float) -> float:
    R = 3963.0
    lat1, lng1 = np.deg2rad(lat1), np.deg2rad(lng1)
    lat2, lng2 = np.deg2rad(lat2), np.deg2rad(lng2)
    return R * arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))


def distance_to_nearest_mrt(row, mrt_df):
    """
    Return:
    - Distance (in km)
    - MRT Code
    """
    lat, lng = row.latitude, row.longitude

    mrt_distances = mrt_df.apply(
        lambda row: great_circle_distance(lat, lng, row.latitude, row.longitude),
        axis=1
    )

    min_mrt_distance = mrt_distances.min()
    min_idx = np.argmin(mrt_distances)

    return min_mrt_distance, mrt_df['code'].iloc[min_idx]


def preprocess_v2(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Same as v1 with additional distance from the apartment
    to the closest existing MRTs.
    """
    mrt_df = pd.read_csv(MRT_DATAFRAME_PATH)
    dataframe = preprocess_v1(dataframe)
    dataframe[['nearest_mrt_dist', 'nearest_mrt_code']] = \
        dataframe.parallel_apply(
            lambda row: distance_to_nearest_mrt(row, mrt_df),
            axis=1,
            result_type="expand"
    )
    return dataframe

import pandas as pd
import numpy as np
from numpy import arccos, cos, sin
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

MRT_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-mrt-existing-stations.csv"
SHOPPING_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-shopping-malls.csv"
SCHOOL_DATAFRAME_PATH = "data/auxiliary-data/auxiliary-data/sg-primary-schools.csv"
POSITION_ATTRS = ['latitude', 'longitude']
TARGET_ATTR = 'monthly_rent'


def is_train_set(dataframe):
    return TARGET_ATTR in dataframe.columns


def preprocess_v1(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the following attributes:
    - rent_approval_date
    - town
    - street name
    - furnished
    - elevation
    - planning_area
    - region
    """
    reg_dates = pd.to_datetime(dataframe['rent_approval_date'], format="%Y-%m")
    dataframe = dataframe\
        .assign(
            year=reg_dates.dt.year,
            month=reg_dates.dt.month,
            flat_type=lambda df: df.flat_type.str.replace('-', ' '),
            floor_area_sqm=lambda df: np.sqrt(df.floor_area_sqm.values),
            block=lambda df: df.town + "-" + df.block,
            street_name=lambda df: df.street_name.str.lower()
        )\
        .drop(columns=[
            'rent_approval_date',
            'furnished',
            'elevation',
            'town',
            'planning_area',
            'region',
        ])

    return dataframe


def great_circle_distance(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = np.pi / 180
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    lat1, lng1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2, lng2 = np.deg2rad(lat2), np.deg2rad(lon2)
    a = np.sin(dlat/2.0) ** 2.0 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2)**2.0)
    return 2 * r * np.arctan2(np.sqrt(a),np.sqrt(1.0 - a))
    

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


def preprocess_v4(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Same as v3 with additional distance from the apartment
    to the closest existing primary schools

    """
    dataframe = preprocess_v3(dataframe)

    school_df = pd.read_csv(SCHOOL_DATAFRAME_PATH).drop_duplicates(POSITION_ATTRS)
    dataframe[['nearest_school_dist', 'nearest_school_name']] = \
        dataframe.parallel_apply(
            lambda row: distance_to_nearest_place(row, school_df, 'name'),
            axis=1,
            result_type="expand"
    )
    dataframe = dataframe.drop(columns=['nearest_school_name'])
    return dataframe

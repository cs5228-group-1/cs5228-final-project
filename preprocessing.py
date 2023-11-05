import pandas as pd
import numpy as np
from numpy import arccos, cos, sin
from pandarallel import pandarallel
from typing import List, Dict
from abc import ABC, abstractmethod
from path import Path


pandarallel.initialize(progress_bar=True)

MRT_DATAFRAME_PATH = "auxiliary-data/auxiliary-data/sg-mrt-existing-stations.csv"
SHOPPING_DATAFRAME_PATH = "auxiliary-data/auxiliary-data/sg-shopping-malls.csv"
SCHOOL_DATAFRAME_PATH = "auxiliary-data/auxiliary-data/sg-primary-schools.csv"
POSITION_ATTRS = ['latitude', 'longitude']
TARGET_ATTR = 'monthly_rent'
CATEGORY_FEATURE_NAMES = [
    'street_name',
    'block',
    'flat_type',
    'flat_model',
    'subzone',
    'nearest_mrt_code',
    'nearest_mall_name',
    'nearest_school_name',
]


def cat_attr_to_id(dataframe: pd.DataFrame) -> List[int]:
    return [
        idx for idx in range(len(dataframe.columns))
        if dataframe.columns[idx] in CATEGORY_FEATURE_NAMES
    ]


def is_train_set(dataframe):
    return TARGET_ATTR in dataframe.columns


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
    - Distance (in meters)
    - Code/Name
    """
    lat, lng = row.latitude, row.longitude

    distances = great_circle_distance(
        lat, lng, df.latitude.values, df.longitude.values)

    min_distance = np.nanmin(distances) * 1000
    min_idx = np.nanargmin(distances)

    return min_distance, df[code_col].iloc[min_idx]


class TransformBase(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ...


class V1(TransformBase):
    """
    Do not consider additional data. Only transform attributes from
    the provided training data.

    Drop the following attributes:
    - rent_approval_date
    - town
    - street name
    - furnished
    - elevation
    - planning_area
    - region
    """

    def __init__(self, cfg):
        super(V1, self).__init__(cfg)

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        reg_dates = pd.to_datetime(
            dataframe['rent_approval_date'], format="%Y-%m")
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


class V2(TransformBase):
    """
    Same as v1 with additional distance from the apartment
    to the closest existing MRTs.
    """

    def __init__(self, cfg):
        super(V2, self).__init__(cfg)
        self.v1 = V1(cfg)
        self.mrt_df = pd.read_csv(
            Path(cfg["datadir"]) / MRT_DATAFRAME_PATH
        ).drop_duplicates(POSITION_ATTRS)

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.v1.apply(dataframe)
        dataframe[['nearest_mrt_dist', 'nearest_mrt_code']] = \
            dataframe.apply(
                lambda row: distance_to_nearest_place(row, self.mrt_df, 'code'),
                axis=1,
                result_type="expand"
        )
        return dataframe


class V3(TransformBase):
    """
    Same as v2 with additional distance from the apartment
    to the closest existing shopping malls.
    """

    def __init__(self, cfg):
        super(V3, self).__init__(cfg)
        self.v2 = V2(cfg)
        self.mall_df = pd.read_csv(
            Path(cfg["datadir"]) / SHOPPING_DATAFRAME_PATH
        ).drop_duplicates(POSITION_ATTRS)

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.v2.apply(dataframe)

        dataframe[['nearest_mall_dist', 'nearest_mall_name']] = \
            dataframe.parallel_apply(
                lambda row: distance_to_nearest_place(row, self.mall_df, 'name'),
                axis=1,
                result_type="expand"
        )
        return dataframe


class V4(TransformBase):
    """
    Same as v3 with additional distance from the apartment
    to the closest existing primary schools

    """

    def __init__(self, cfg):
        super(V4, self).__init__(cfg)
        self.v3 = V3(cfg)
        self.school_df = pd.read_csv(
            Path(cfg["datadir"]) / SCHOOL_DATAFRAME_PATH
        ).drop_duplicates(POSITION_ATTRS)

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.v3.apply(dataframe)

        dataframe[['nearest_school_dist', 'nearest_school_name']] = \
            dataframe.parallel_apply(
                lambda row: distance_to_nearest_place(row, self.school_df, 'name'),
                axis=1,
                result_type="expand"
        )
        return dataframe

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ('RH', 'T', 'U', 'V')
FOLDS = (('2024-05-19', '2024-06-02'), ('2024-06-02', '2024-06-16'),
         ('2024-06-16', '2024-07-01'))


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    times: pd.DatetimeIndex
    static: dict


def load_dataset(root='data'):
    """Load only official training records and static fields; never load test for fitting."""
    root = Path(root)
    with np.load(root / 'train.npz', allow_pickle=False) as src:
        x, y, stamps = (src[k] for k in ('X', 'Y', 'times'))
    with np.load(root / 'static.npz', allow_pickle=False) as src:
        static = dict(src)
    if stamps.shape != (2935,) or stamps.dtype.kind not in 'US':
        raise ValueError('Unexpected timestamp shape or dtype')
    text = pd.Series(stamps.astype(str))
    if not text.str.fullmatch(r'[0-9]{10}').all():
        raise ValueError('Expected YYYYMMDDHH timestamps')
    times = pd.DatetimeIndex(pd.to_datetime(text, format='%Y%m%d%H', utc=True))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Expected strictly increasing unique timestamps')
    arrays = [(x, (2935,17,13,4)), (y, (2935,70,35,4))]
    arrays += [(static[k], shape) for k, shape in {'dem':(70,35), 'gfs_lats':(17,),
        'gfs_lons':(13,), 'wrf_lats':(70,), 'wrf_lons':(35,)}.items()]
    for array, shape in arrays:
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f'Expected finite float32 array of shape {shape}')
    if tuple(static['vars']) != CHANNELS:
        raise ValueError('Unexpected channel order')
    if not (np.diff(static['gfs_lats']) < 0).all() or not all(
            (np.diff(static[k]) > 0).all() for k in ('gfs_lons','wrf_lats','wrf_lons')):
        raise ValueError('Unexpected geographic orientation')
    return Dataset(x, y, times, static)


def split_indices(times, start, end):
    train = times < pd.Timestamp(start, tz='UTC')
    valid = (times >= pd.Timestamp(start, tz='UTC')) & (times < pd.Timestamp(end, tz='UTC'))
    if not train.any() or not (valid & (times.hour % 3 == 0)).any():
        raise ValueError('Empty training or scored validation split')
    return np.flatnonzero(train), np.flatnonzero(valid)

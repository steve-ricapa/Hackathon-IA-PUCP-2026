from dataclasses import dataclass

import numpy as np
import pandas as pd


def temporal_windows(x, times, lags=(2, 1, 0)):
    """Exact causal lags, fallback to current X; True in the returned mask means missing."""
    if len(x) != len(times) or times.has_duplicates or times.hasnans:
        raise ValueError('Expected aligned, valid, unique timestamps')
    if not lags or any(lag < 0 or int(lag) != lag for lag in lags) or 0 not in lags:
        raise ValueError('Expected nonnegative integer lags including 0')
    idx = np.stack([times.get_indexer(times - pd.Timedelta(hours=lag)) for lag in lags], axis=1)
    missing = idx < 0
    return x[np.where(missing, np.arange(len(x))[:, None], idx)], missing


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_sd: np.ndarray
    clim: np.ndarray
    anom_sd: np.ndarray

    @classmethod
    def fit(cls, x_train, y_train, hours_train):
        """Pass only the training side of a fold. Match starter FP32 reductions."""
        if len(x_train) != len(y_train) or len(y_train) != len(hours_train):
            raise ValueError('Training arrays must be aligned')
        if set(hours_train) != set(range(24)):
            raise ValueError('Fit requires climatology support for each hour')
        clim = np.stack([y_train[hours_train == h].mean(0) for h in range(24)])
        scale = (y_train - clim[hours_train]).std(axis=(0,1,2))
        x_sd = x_train.std(axis=(0,1,2))
        if not (np.isfinite(scale).all() and np.isfinite(x_sd).all()
                and (scale > 0).all() and (x_sd > 0).all()):
            raise ValueError('Invalid feature/target scales')
        return cls(x_train.mean(axis=(0,1,2)), x_sd, clim, scale)

    def inputs(self, windows):
        return (windows - self.x_mean) / self.x_sd

    def targets(self, targets, hours):
        return (targets - self.clim[hours]) / self.anom_sd

    def physical(self, anomalies, hours):
        return self.clim[hours] + anomalies * self.anom_sd

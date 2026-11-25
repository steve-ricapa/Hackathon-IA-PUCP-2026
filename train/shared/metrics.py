import numpy as np

OFFICIAL_SCALE = np.array([25.360, 2.0652, 2.6776, 2.0876])


def score_mask(times):
    return np.asarray(times.hour % 3 == 0)


def rmse(pred, truth):
    if pred.shape != truth.shape or pred.ndim != 4 or pred.shape[-1] != 4 or len(pred) == 0:
        raise ValueError('Expected pair of aligned, nonempty arrays (time, row, column, 4)')
    diff = np.subtract(pred, truth, dtype=np.float64)
    if not np.isfinite(diff).all():
        raise ValueError('Non-finite prediction or truth')
    return np.sqrt(np.mean(diff**2, axis=(0,1,2)))


def evaluate(pred, truth, times, climatology):
    """In physical units. `climatology` is aligned to the same data as pred/truth."""
    mask = score_mask(times)
    err, ref = rmse(pred[mask], truth[mask]), rmse(climatology[mask], truth[mask])
    fixed = 1 - err / OFFICIAL_SCALE
    return dict(hours=int(mask.sum()), rmse=err.tolist(), rmse_clim=ref.tolist(),
                official=float(fixed.mean()), official_by_channel=fixed.tolist(),
                skill_clim=float(1 - (err / ref).mean()) if (ref > 0).all() else None)

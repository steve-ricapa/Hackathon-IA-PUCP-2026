import json
from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS


def aligned_predictions(paths):
    """Return equal-fold validation predictions; reject differences in time, grid or channel order."""
    keys = ('times', 'channels', 'units', 'wrf_lats', 'wrf_lons', 'scored', 'start', 'end_exclusive')
    arrays, metadata = [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            current = {k: src[k] for k in keys}
            pred = src['prediction']
        if (tuple(current['channels']) != CHANNELS or
                tuple(current['units']) != ('%', 'K', 'm/s', 'm/s') or
                pred.shape != (len(current['times']),70,35,4) or not np.isfinite(pred).all()):
            raise ValueError(f'Invalid validation prediction artifact: {path}')
        if metadata is None:
            metadata = current
        elif any(not np.array_equal(metadata[k], current[k]) for k in keys):
            raise ValueError(f'Validation fold/artifact not aligned: {path}')
        arrays.append(pred)
    if not arrays:
        raise ValueError('No validation predictions provided')
    return metadata, arrays


class Experiment:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)  # do not collide with another agent's run

    def json(self, name, data):
        dest = self.path / f'{name}.json'
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        tmp.replace(dest)

    def epoch(self, row):
        with (self.path / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def checkpoint(self, model, epoch, metric):
        tmp = self.path / 'best.tmp'
        torch.save(dict(state_dict={k: v.detach().cpu() for k,v in model.state_dict().items()},
                        epoch=epoch, metric=metric), tmp)
        tmp.replace(self.path / 'best.pt')

    def predictions(self, name, pred, times, static, scored, start, end):
        if pred.shape != (len(times),70,35,4) or not np.isfinite(pred).all():
            raise ValueError('Invalid validation prediction array')
        dest = self.path / f'{name}.npz'
        tmp = dest.with_suffix('.tmp')
        with tmp.open('wb') as file:
            np.savez_compressed(file, prediction=pred.astype(np.float32, copy=False),
                times=np.asarray(times.strftime('%Y%m%d%H'), dtype='U10'),
                channels=np.asarray(CHANNELS), units=np.array(['%', 'K', 'm/s', 'm/s']),
                wrf_lats=static['wrf_lats'], wrf_lons=static['wrf_lons'], scored=scored,
                start=start, end_exclusive=end)
        tmp.replace(dest)

    def plot(self, history):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [r['epoch'] for r in history]
        fig, ax = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for k in ('train_mse', 'val_mse'):
            ax[0].plot(epochs, [r[k] for r in history], label=k)
        ax[0].set(xlabel='Epoch', ylabel='Normalized anomaly MSE')
        ax[0].legend()
        ax[1].plot(epochs,[r['cnn']['official'] for r in history])
        ax[1].set(xlabel='Epoch', ylabel='Official score (3h UTC)')
        fig.savefig(self.path / 'curves.png', dpi=140)
        plt.close(fig)

import numpy as np


def starter_grid(static):
    """Grid for torch.grid_sample with align_corners=True on the regular official GFS grid."""
    lat, lon = np.meshgrid(static['wrf_lats'], static['wrf_lons'], indexing='ij')
    y, x = static['gfs_lats'], static['gfs_lons']
    for axis in (y, x):
        if not np.allclose(np.diff(axis), axis[1] - axis[0]):
            raise ValueError('Normalized starter grid requires regular GFS axes')
    return np.stack(((lon-x[0])/(x[-1]-x[0])*2-1, (lat-y[0])/(y[-1]-y[0])*2-1), -1)[None]


def bilinear(x, static):
    """Coordinate-based interpolation in WRF orientation, clamping out-of-domain queries."""
    def indices(axis, query):
        q = np.clip(query, axis[0], axis[-1])
        i = np.searchsorted(axis, q, side='right').clip(1, len(axis)-1)-1
        return i, (q-axis[i])/(axis[i+1]-axis[i])

    i, dy = indices(static['gfs_lats'][::-1], static['wrf_lats'])
    j, dx = indices(static['gfs_lons'], static['wrf_lons'])
    dy, dx = dy[None,:,None,None], dx[None,None,:,None]
    x = x[:, ::-1]

    def get(row, col):
        return x[:, row[:,None], col[None,:], :]

    return ((1-dy)*((1-dx)*get(i,j)+dx*get(i,j+1)) + dy*((1-dx)*get(i+1,j)+dx*get(i+1,j+1)))

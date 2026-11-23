"""
Hackathon IA PUCP 2026 · Pronóstico GFS actual para tu producto.

Descarga el pronóstico GFS más reciente desde Open-Meteo (gratis, sin API key) en el
MISMO formato que el input X de la competencia:

    X: (horas, 17, 13, 4)  ->  RH (%), T (K), U (m/s), V (m/s)
    grilla 0.25°: lat -7.25 a -11.25 (fila 0 = norte), lon -79.0 a -76.0 (columna 0 = oeste)
    times: 'YYYYMMDDHH' en UTC

Uso:
    from gfs_forecast import get_gfs_forecast, make_windows
    X, times = get_gfs_forecast(forecast_hours=48)      # incluye 2 horas pasadas para la ventana t-2..t
    Xw, tw = make_windows(X, times)                     # (horas, 3, 17, 13, 4) = [t-2, t-1, t]
    Xp, tp = get_gfs_forecast(start_date="2026-07-01", end_date="2026-07-07")  # fechas pasadas
    # ...normaliza igual que en tu entrenamiento y pásalo a tu modelo

Requisitos: pip install numpy requests

Límites del plan gratis de Open-Meteo: ~600 llamadas/min, 5,000/hora, 10,000/día.
Cada descarga de la grilla completa cuenta como ~221 llamadas (una por punto).
=> En tu producto, NO descargues en cada request del usuario: guarda el resultado
   (cache) y refréscalo cada 1–6 horas. El GFS se actualiza 4 veces al día.

Regla de la hackathon: esta función es SOLO para el producto (datos actuales).
No la uses para bajar más data de entrenamiento ni para la submission de Kaggle.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests

API = "https://api.open-meteo.com/v1/forecast"
API_HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
VARS = ["RH", "T", "U", "V"]

LATS = np.round(np.arange(-7.25, -11.25 - 1e-9, -0.25), 2)  # 17, fila 0 = norte
LONS = np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2)    # 13, columna 0 = oeste
assert len(LATS) == 17 and len(LONS) == 13


def _fetch_chunk(lats, lons, past_hours, forecast_hours, start_date=None, end_date=None, retries=5):
    params = {
        "latitude": ",".join(str(a) for a in lats),
        "longitude": ",".join(str(b) for b in lons),
        "hourly": HOURLY,
        "models": "gfs_global",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    if start_date:   # fechas pasadas: archivo histórico del mismo pronóstico GFS
        url = API_HIST
        params.update(start_date=start_date, end_date=end_date or start_date)
    else:            # pronóstico actual
        url = API
        params.update(past_hours=past_hours, forecast_hours=forecast_hours)
    for intento in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            js = r.json()
            return js if isinstance(js, list) else [js]
        if r.status_code == 429:  # límite de la API: espera y reintenta
            time.sleep(30 * (intento + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("Open-Meteo no respondió después de varios intentos (¿límite de la API?).")


def get_gfs_forecast(forecast_hours=48, past_hours=2, start_date=None, end_date=None,
                     chunk=50, cache_path=None, cache_max_age_h=1.0):
    """
    Devuelve (X, times):
      X     : np.ndarray float32 (H, 17, 13, 4) con RH (%), T (K), U, V (m/s)
      times : np.ndarray str (H,) 'YYYYMMDDHH' en UTC

    forecast_hours : horas hacia adelante desde ahora (máx. ~384).
    past_hours     : horas pasadas incluidas (2 = suficiente para la ventana t-2, t-1, t).
    start_date, end_date : 'YYYY-MM-DD'. Si se dan, descarga esas fechas PASADAS en vez del
                     pronóstico actual (p. ej. para comparar en tu producto). Solo para el producto.
    cache_path     : si se da (p. ej. 'gfs_cache.npz'), reusa la descarga si tiene menos de cache_max_age_h horas.
    """
    if cache_path and not start_date and os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h < cache_max_age_h:
            d = np.load(cache_path)
            return d["X"], d["times"]

    LA, LO = np.meshgrid(LATS, LONS, indexing="ij")
    pts = list(zip(LA.ravel().tolist(), LO.ravel().tolist()))
    series, times = {}, None
    for i in range(0, len(pts), chunk):
        sub = pts[i:i + chunk]
        js = _fetch_chunk([a for a, _ in sub], [b for _, b in sub], past_hours, forecast_hours,
                          start_date, end_date)
        if start_date and i + chunk < len(pts):
            time.sleep(5)   # rangos largos pesan más en el límite de la API
        for (a, b), j in zip(sub, js):
            h = j["hourly"]
            times = h["time"]
            ws = np.array(h["wind_speed_10m"], dtype=float)
            wd = np.deg2rad(np.array(h["wind_direction_10m"], dtype=float))
            series[(a, b)] = np.stack([
                np.array(h["relative_humidity_2m"], dtype=float),
                np.array(h["temperature_2m"], dtype=float) + 273.15,   # °C -> K
                -ws * np.sin(wd),                                       # U (oeste -> este)
                -ws * np.cos(wd),                                       # V (sur -> norte)
            ], axis=1)

    X = np.stack([np.stack([series[(LATS[r], LONS[c])] for c in range(13)], axis=1)
                  for r in range(17)], axis=1).astype(np.float32)       # (H, 17, 13, 4)
    t = np.array([s.replace("-", "").replace("T", "")[:10] for s in times])

    # Horas incompletas (pasa a veces en el último tramo del pronóstico): se descartan
    ok = ~np.isnan(X).any(axis=(1, 2, 3))
    X, t = X[ok], t[ok]

    if cache_path and not start_date:
        np.savez_compressed(cache_path, X=X, times=t)
    return X, t


def make_windows(X, times, k=3):
    """
    Arma ventanas [t-(k-1), ..., t] como en el starter: (H, k, 17, 13, 4).
    Las primeras horas repiten la más cercana disponible.
    """
    idx = np.arange(len(X))
    win = np.stack([X[np.clip(idx - (k - 1 - j), 0, None)] for j in range(k)], axis=1)
    return win, times


if __name__ == "__main__":
    X, t = get_gfs_forecast(forecast_hours=24)
    print("X:", X.shape, "| desde", t[0], "hasta", t[-1], "UTC")
    for c, v in enumerate(VARS):
        print(f"  {v}: min {X[..., c].min():7.2f}  media {X[..., c].mean():7.2f}  max {X[..., c].max():7.2f}")
    print("Ahora (UTC):", datetime.now(timezone.utc).strftime("%Y%m%d%H"))

"""
Hackathon IA PUCP 2026 · Ejemplo mínimo de producto, de punta a punta.

    GFS actual (Open-Meteo)  ->  tu modelo de downscaling  ->  algo útil para un usuario

Este ejemplo hace un "alerta de heladas" muy simple para las próximas 24 horas:
  1. Descarga el pronóstico GFS actual con gfs_forecast.py (grilla 17x13).
  2. Lo pasa por `predecir()`, que devuelve mapas de 3 km (70x35) de RH, T, U, V.
  3. Marca las zonas donde la temperatura baja de 0 °C y guarda un mapa en salidas/.

IMPORTANTE: `predecir()` trae un MODELO DE RELLENO (interpolación + corrección por
altura), solo para que el ejemplo corra. NO es un modelo válido para la hackathon.
Reemplázalo por TU modelo de Kaggle, con los mismos pesos de tu submission final.

Uso:
    pip install -r requirements.txt
    python app/ejemplo_producto.py
    python app/ejemplo_producto.py --desde 2026-07-01 --hasta 2026-07-02   # fechas pasadas

Por defecto lee data/static.npz (se baja de la competencia en Kaggle, pestaña Data)
y guarda el mapa en app/salidas/.
"""
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS, get_gfs_forecast, make_windows

VARS = ["RH", "T", "U", "V"]
APP = Path(__file__).resolve().parent          # carpeta app/
RAIZ = APP.parent                                # raíz del repositorio


# ---------------------------------------------------------------------------
# 1. TU MODELO VA AQUÍ
# ---------------------------------------------------------------------------
def predecir(Xw, static):
    """
    Entrada:
      Xw     : (N, 3, 17, 13, 4) ventanas [t-2, t-1, t] del GFS, en unidades físicas
               (RH %, T K, U m/s, V m/s). Fila 0 = norte.
      static : contenido de static.npz (dem, wrf_lats, wrf_lons, ...)
    Salida:
      Y      : (N, 70, 35, 4) mapas a 3 km, mismas unidades. FILA 0 = SUR (como en Kaggle).

    Reemplaza este cuerpo por tu modelo. Ejemplo con PyTorch:
        modelo = MiRed(); modelo.load_state_dict(torch.load("models/mi_modelo.pt"))
        Xn = normalizar(Xw)            # igual que en tu entrenamiento
        Y = desnormalizar(modelo(Xn))  # (N, 70, 35, 4)
    """
    # --- MODELO DE RELLENO (no usar en la entrega) ---
    Xt = Xw[:, -1]                                        # solo la hora t
    Y = _interpolar_a_wrf(Xt, static["wrf_lats"], static["wrf_lons"])
    dem = static["dem"].astype(np.float32)
    dem_suave = _suavizar(dem, k=9)                       # relieve "visto" a baja resolución
    Y[..., 1] -= 0.0065 * (dem - dem_suave)               # T baja ~6.5 °C por km extra de altura
    return Y


# ---------------------------------------------------------------------------
# 2. LO QUE EL PRODUCTO HACE CON LA SALIDA DEL MODELO
# ---------------------------------------------------------------------------
def alerta_heladas(Y, times, umbral_c=0.0):
    """Para cada píxel: ¿habrá helada en el periodo? y la hora (Lima) de la mínima."""
    T_c = Y[..., 1] - 273.15                               # (N, 70, 35)
    tmin = T_c.min(axis=0)
    hora_min = np.asarray(times)[T_c.argmin(axis=0)]
    helada = tmin < umbral_c
    return tmin, hora_min, helada


def a_hora_lima(t_utc):
    return (datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)).strftime("%a %d/%m %H:00")


def guardar_mapa(tmin, helada, static, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats, lons = static["wrf_lats"], static["wrf_lons"]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    fig, ax = plt.subplots(1, 2, figsize=(9, 8), constrained_layout=True)
    im = ax[0].imshow(tmin, origin="lower", extent=ext, cmap="coolwarm")   # origin=lower: fila 0 = sur
    ax[0].set_title("Temperatura mínima (°C)")
    fig.colorbar(im, ax=ax[0], shrink=0.7)
    ax[1].imshow(static["dem"], origin="lower", extent=ext, cmap="gray")
    ax[1].contourf(np.linspace(ext[0], ext[1], tmin.shape[1]), np.linspace(ext[2], ext[3], tmin.shape[0]),
                   helada.astype(float), levels=[0.5, 1.5], colors=["#4aa3ff"], alpha=0.6)
    ax[1].set_title("Zonas con helada (azul) sobre el relieve")
    for a in ax:
        a.set_xlabel("Longitud"); a.set_ylabel("Latitud")
    fig.suptitle(titulo)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _interpolar_a_wrf(Xt, wrf_lats, wrf_lons):
    """Bilineal por coordenadas: (N, 17, 13, 4) GFS -> (N, 70, 35, 4) WRF (fila 0 = sur)."""
    lat_g = GFS_LATS[::-1]                     # pasar a sur -> norte (creciente)
    Xs = Xt[:, ::-1]
    fi = np.interp(wrf_lats, lat_g, np.arange(len(lat_g)))
    fj = np.interp(wrf_lons, GFS_LONS, np.arange(len(GFS_LONS)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat_g) - 2); di = (fi - i0)[None, :, None, None]
    j0 = np.clip(np.floor(fj).astype(int), 0, len(GFS_LONS) - 2); dj = (fj - j0)[None, None, :, None]
    A = Xs[:, i0][:, :, j0]; B = Xs[:, i0][:, :, j0 + 1]
    C = Xs[:, i0 + 1][:, :, j0]; D = Xs[:, i0 + 1][:, :, j0 + 1]
    return ((A * (1 - dj) + B * dj) * (1 - di) + (C * (1 - dj) + D * dj) * di).astype(np.float32)


def _suavizar(a, k=9):
    p = k // 2
    ap = np.pad(a, p, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            out += ap[di:di + a.shape[0], dj:dj + a.shape[1]]
    return out / (k * k)


def main():
    ap = argparse.ArgumentParser(description="Ejemplo de producto: alerta de heladas")
    ap.add_argument("--static", default=str(RAIZ / "data" / "static.npz"), help="ruta a static.npz (de Kaggle)")
    ap.add_argument("--horas", type=int, default=24, help="horas de pronóstico hacia adelante")
    ap.add_argument("--desde", help="fecha pasada YYYY-MM-DD (opcional)")
    ap.add_argument("--hasta", help="fecha pasada YYYY-MM-DD (opcional)")
    a = ap.parse_args()

    if not os.path.exists(a.static):
        raise SystemExit(f"No encuentro {a.static}. Baja la data de Kaggle a la carpeta data/ (ver README).")
    static = dict(np.load(a.static))
    if a.desde:
        X, times = get_gfs_forecast(start_date=a.desde, end_date=a.hasta or a.desde)
    else:
        X, times = get_gfs_forecast(forecast_hours=a.horas, cache_path=str(APP / "gfs_cache.npz"))
    Xw, times = make_windows(X, times)
    Y = predecir(Xw, static)
    assert Y.shape[1:] == (70, 35, 4), f"predecir() debe devolver (N, 70, 35, 4), devolvió {Y.shape}"

    tmin, hora_min, helada = alerta_heladas(Y, times)
    os.makedirs(APP / "salidas", exist_ok=True)
    ruta = str(APP / "salidas" / "alerta_heladas.png")
    periodo = f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])} (hora de Lima)"
    guardar_mapa(tmin, helada, static, f"Alerta de heladas · {periodo}", ruta)

    print(f"Periodo: {periodo}")
    print(f"Temperatura mínima en el dominio: {tmin.min():.1f} °C")
    print(f"Zonas con helada: {100 * helada.mean():.0f}% del territorio")
    if helada.any():
        i, j = np.unravel_index(tmin.argmin(), tmin.shape)
        print(f"Punto más frío: lat {static['wrf_lats'][i]:.2f}, lon {static['wrf_lons'][j]:.2f}, "
              f"{static['dem'][i, j]:.0f} m, {tmin[i, j]:.1f} °C el {a_hora_lima(hora_min[i, j])}")
    print("Mapa guardado en", ruta)


if __name__ == "__main__":
    main()

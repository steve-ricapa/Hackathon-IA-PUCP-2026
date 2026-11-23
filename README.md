# Hackathon IA PUCP 2026 · Repositorio base

**DSC PUCP × Grupo de Inteligencia Artificial PUCP (IA-PUCP)**

Convierte el pronóstico global del clima (GFS, ~27 km) en mapas de alta resolución (3 km) de humedad, temperatura y viento para la Cordillera Blanca, y llévalo a un producto que le sirva a alguien real.

> **Lleva tu modelo más allá de la métrica.**

Este repositorio es la **plantilla de tu equipo**: tiene la estructura, el starter del modelo y un ejemplo de producto que ya corre. Las reglas completas están en las **bases** y el detalle de cómo participar, en la **Guía del participante**. Si hay una contradicción, prevalecen las bases.

## 0. Crea el repositorio de tu equipo

1. Arriba a la derecha, haz clic en **Use this template → Create a new repository**.
2. Créalo en la cuenta de un integrante, **público**, con el nombre que quieran.
3. Agrega a los demás integrantes como colaboradores (*Settings → Collaborators*).
4. Clónalo y trabaja ahí. La URL de ese repositorio es tu entregable B.

## Estructura

```text
├── README.md               ← esta guía (reemplázala por la de tu producto al final)
├── REPORTE_PRODUCTO.md     ← reporte que completa tu equipo
├── requirements.txt
├── data/                   ← aquí bajas la data de Kaggle (no se sube a GitHub)
├── train/
│   └── starter_notebook.ipynb   ← de la data oficial al submission.csv (score ≈ 0.40)
├── models/                 ← pesos de tu modelo final
└── app/                    ← tu producto
    ├── gfs_forecast.py     ← descarga el pronóstico GFS en el formato de tu modelo
    └── ejemplo_producto.py ← ejemplo de punta a punta: GFS → modelo → alerta de heladas
```

## 1. Baja la data (desde Kaggle)

La data **no está en este repositorio** (pesa más de lo que GitHub acepta). Bájala desde la pestaña *Data* de la competencia o con la API de Kaggle, desde la raíz del repositorio:

```bash
pip install kaggle
kaggle competitions download -c hackathon-ia-pucp-2026
unzip hackathon-ia-pucp-2026.zip -d data/
```

| Archivo | Contenido |
|---|---|
| `train.npz` | `X` (2935, 17, 13, 4) GFS · `Y` (2935, 70, 35, 4) WRF · `times` — 18 feb a 30 jun 2024 |
| `test.npz` | `X` (792, 17, 13, 4) · `times` — 2 jul a 4 ago 2024 (sin `Y`: eso es lo que predices) |
| `static.npz` | `dem` (70, 35) en metros y las coordenadas de ambas grillas |
| `sample_submission.csv` | Formato de envío, lleno con la climatología (score 0) |

**Convenciones**

- Variables, en este orden: `RH` (%), `T` (K), `U` (m/s), `V` (m/s).
- GFS 17×13: **fila 0 = norte**. WRF y DEM 70×35: **fila 0 = sur**. Columna 0 = oeste en ambas.
- `times` en formato `YYYYMMDDHH` y en **UTC** (hora de Perú = UTC − 5).
- Valida con un corte temporal (por ejemplo, las últimas 2 semanas de junio). **Nunca** con un split aleatorio.

## 2. Entrena tu modelo (`train/`)

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook train/starter_notebook.ipynb
```

El starter carga la data de `data/`, valida con la métrica oficial, entrena una CNN chica con el relieve, genera `submission.csv` y guarda los pesos en `models/modelo.pt`. También corre en Kaggle (pestaña *Code* de la competencia, con *GPU T4*) o en Google Colab.

Úsalo como punto de partida: tu versión final debe quedar en `train/` como un **notebook reproducible** que, corrido de principio a fin solo con la data oficial, entrene tu modelo y genere el `submission.csv` de tu submission final.

## 3. Construye tu producto (`app/`)

Tu producto usa **el mismo modelo** con el que compites en Kaggle, alimentado con el pronóstico GFS de Open-Meteo. Alrededor puedes agregar lo que quieras: LLMs, traducción, alertas, otras fuentes de datos.

```python
from gfs_forecast import get_gfs_forecast, make_windows

X, times = get_gfs_forecast(forecast_hours=48, cache_path="gfs_cache.npz")
# X: (horas, 17, 13, 4) = RH (%), T (K), U, V (m/s) · times en UTC
Xw, tw = make_windows(X, times)   # (horas, 3, 17, 13, 4) = [t-2, t-1, t]
# normaliza igual que en tu entrenamiento y pásalo a tu modelo

# Fechas pasadas (opcional, solo para el producto):
Xp, tp = get_gfs_forecast(start_date="2026-07-01", end_date="2026-07-07")
```

Prueba el ejemplo de punta a punta (necesita `data/static.npz`):

```bash
python app/ejemplo_producto.py
```

Genera `app/salidas/alerta_heladas.png`. Luego reemplaza la función `predecir()` por tu modelo, cargando los pesos de `models/`, y construye tu producto sobre esa base (o reemplázalo por completo).

**Usa caché.** Cada descarga de la grilla completa cuenta como ~221 llamadas al plan gratis de Open-Meteo (5,000 por hora). Guarda el resultado y refréscalo cada 1 a 6 horas; no descargues en cada consulta del usuario.

**Regla de datos:** `gfs_forecast.py` es solo para el producto. Para entrenar y para la submission de Kaggle solo se puede usar la data oficial de la competencia.

## 4. Lo que debe tener tu repositorio al entregar

- [ ] `README.md` reescrito para tu producto: qué hace y para quién, cómo instalarlo y ejecutarlo en pocos pasos, el link al video pitch y los integrantes del equipo.
- [ ] **Un notebook reproducible de entrenamiento** en `train/` que, corrido de principio a fin solo con la data oficial, entrene tu modelo y genere el `submission.csv` de tu submission final. Fija las semillas, usa rutas relativas y déjalo con las salidas guardadas. Debe poder correr en Kaggle o Colab. Con nombre `submission_notebook.ipynb`
- [ ] **Los pesos del modelo** en `models/`. Guarda solo los pesos, no el estado del optimizador. Si aun así pasan de 100 MB (el límite de GitHub), súbelos a Google Drive o Hugging Face, con acceso para cualquiera con el link, y pon el link en el README.
- [ ] El código del producto en `app/`, que carga **esos mismos pesos**.
- [ ] `requirements.txt` con las versiones de las librerías que usaste.
- [ ] Ninguna clave de API en el código: usa variables de entorno.
- [ ] `REPORTE_PRODUCTO.md` completo (ver abajo).

### El reporte del producto

Completa `REPORTE_PRODUCTO.md` respondiendo sus 6 preguntas, de forma breve:

1. ¿Qué modelo o arquitectura usaron?
2. ¿Para quién es y qué problema le resuelve?
3. ¿Cuál es la propuesta de valor del producto?
4. ¿Por qué tu producto es diferente a algo usual?
5. ¿Cómo usa el producto la salida de tu modelo?
6. Limitaciones y próximos pasos.

El jurado lo lee durante la preselección: es la forma más rápida de que entienda tu solución antes de ver el video.

## Fechas clave (hora de Lima)

| Hito | Fecha |
|---|---|
| Cierre de inscripciones | lun 7 dic, 23:59 |
| **Cierre de Kaggle y entregables** | **mar 8 dic, 23:59** |
| Anuncio de finalistas | jue 10 dic |
| Final presencial | sáb 12 dic, 2:00–5:00 p. m., auditorio de FACI (PUCP) |

## Referencia

Carhuancho, C., Villanueva, E., Yarlequé, C., Principe, R. E., & Castromonte, M. (2025). *Generating high-resolution climate data in the Andes using artificial intelligence: A lightweight alternative to the WRF model.* Artificial Intelligence in Geosciences, 6, 100143. https://doi.org/10.1016/j.aiig.2025.100143

La data de esta competencia fue reprocesada para el reto: no compares tus resultados con las métricas del paper.

## Contacto

DSC PUCP · dsc.pucp@gmail.com · Daniel Malca, organizador · daniel.malca@pucp.edu.pe

Dudas sobre la data o las reglas: en el grupo de participantes.

# R8 — adaptación del starter

`READY` verificado; exploración hasta 00:12:15 UTC, confirmación hasta 01:42:15 UTC y entrega final compartida a las 03:12:15 UTC del 26 de noviembre de 2026. Solo C0–C2 (ya explorados), sin C3 ni test. Se usa el starter original, el mismo calendario AdamW/OneCycle y los controles de A, no se entrenan controles comunes duplicados. `main` es la copia online del control; `ema=True` describe la producción de sombras adicionales y no significa que `main` sea una sombra.

| Hipótesis | Control | Plan |
|---|---|---|
| B1 | Starter de A, época 21 | Semivida de 60 días, pesos medios 1 en todo el entrenamiento. 30/90 solo si se observa señal. |
| B2 | Decoder con datos uniformes | Decoder de época 21, AdamW reiniciado y LR 1e−4, 100/300 actualizaciones, ventana de 30 días. 60 días solo si pasa 30. |
| B3 | Head compartida de 4 canales | Tronco congelado y lectura por píxel, lambda 0.1/1.0. Head compartida: 272 parámetros entrenables; localizada: 39,472. Capacidad y localización no se aíslan por completo en este primer contraste. |
| B4 | B temporal y B3 independientes | Solo si las dos familias pasan; no combinar ponderación y adaptación reciente. |

Comprobación CPU con los pesos reales de A: el adaptador inicializado a cero conserva exactamente la salida del starter en los tres cortes; el rescore de cada control reproduce el manifiesto. Para B1-60, rangos de pesos C0/C1/C2: 0.631–1.535, 0.566–1.619 y 0.513–1.725. Tamaños efectivos descriptivos: 1514/1616, 1779/1951 y 2014/2263 horas. No son tamaños de muestra independientes.

### Screening B1 (semilla 0)

`B1-60` obtuvo **0.444972** frente al control **0.441886** (+0.003086). Delta C0/C1/C2: −0.000023 / −0.001042 / +0.010523. RMSE (RH/T/U/V): 10.20239 / 0.96650 / 1.67471 / 1.51218, frente a 10.14010 / 0.97970 / 1.69356 / 1.51504 del control. La ganancia agrupada sobrevive al excluir cualquier semana, pero apenas (+0.000150) en el peor caso y no cumple dos cortes favorables. No se promueve ni se atribuye el resultado a una mejora estacional general. Esta moderada ganancia en C2 acompañada de pérdidas en C0/C1 justifica los dos contrastes 30/90 para distinguir una adaptación insuficiente de una excesiva. Se verificó también la inferencia del checkpoint C0 en CPU contra la predicción guardada.

### Screening B2, 100 actualizaciones (semilla 0)

Los últimos 30 días obtuvieron **0.439881** contra **0.439556** del decoder adaptado con historial completo: +0.000324. Los tres cortes avanzaron frente al control de actualizaciones extra, pero C2 apenas (+0.000002). Ambas ramas retroceden frente al starter sin ajuste (0.441886). Se publicó `CANDIDATE_READY` para confirmar el contraste, no como recomendación de integración. La comparación principal de B2 es contra las mismas actualizaciones, y la recuperación frente al starter es una segunda pregunta.

R7 está congelada. `R7_HANDOFF.json` confirma hashes: C0 es R4; en C1/C2 solo V incorpora `R4 + 0.25*(clock-lags − R4)`. U, T y RH se conservan. Si A usa el artefacto tal cual (peso externo 1), no se reaplica 0.25.

## Reproducción

Desde este worktree, en Bash y sin lanzar un segundo escritor simultáneo:

```bash
PY=../Hackathon-IA-PUCP-2026/.venv/Scripts/python.exe
"$PY" -B -m unittest train.agent_b.round8.test_round8
"$PY" -B -m train.agent_b.round8.audit
OMP_NUM_THREADS=2 "$PY" -B -u -m train.agent_b.round8.campaign --watch
"$PY" -B -m train.agent_b.round8.run B1-60 C0 --seed 0 --output '<directorio nuevo indicado por A>'
"$PY" -B -m train.agent_b.round8.replay B1-60 C0 --seed 0 --output '<directorio nuevo indicado por A>'
```

Las dos últimas interfaces son de desarrollo y no habilitan C3. El estado y los artefactos están en la raíz privada `round8/agent_b` del repositorio original. El lanzamiento no equivale a tener un candidato. Si no hay controles disponibles se espera sin GPU. Las métricas se reconstruyen en unidades físicas con clipping de RH al puntuar, no al exportar.

Referencias: [DL4DS, bloque localizado](https://doi.org/10.1017/eds.2022.26) inspira la compresión seguida de parámetros locales; no se trasladan sus resultados de calidad del aire. [Kassis et al. (2026)](https://doi.org/10.48550/arXiv.2609.00065) para revisión metodológica.

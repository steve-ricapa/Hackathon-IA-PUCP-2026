# R7 — downscaling estadístico no lineal

## Diseño y literatura

- [Alessi y DeGaetano (2021)](https://doi.org/10.1002/met.1976) comparan regresión estadística y downscaling dinámico de pronósticos de corto plazo del noreste de EE. UU. Motivación: evaluar MOS donde la relación depende del lugar; sus resultados no son un benchmark transferible de viento andino.
- [Glahn y Lowry (1972)](https://doi.org/10.1175/1520-0450%281972%29011%3C1203%3ATUOMOS%3E2.0.CO%3B2) motivan la calibración con predictores de modelos numéricos. Aquí se aprende directamente contra WRF, no contra estaciones y no contra residuos del ensemble.
- [Micchelli y Pontil (2005)](https://pubmed.ncbi.nlm.nih.gov/15563752/) encuadran kernels vectoriales. Usamos el caso separable `k(x,x') I`: todas las salidas comparten una factorización, sin imponer covarianza entre canales ni reducir el rango de salida a un EOF. El producto del RBF meteorológico con el kernel cíclico permite que la similitud meteorológica dependa de la fase diurna.
- El MOS ajusta coeficientes por píxel con penalización modulada por elevación, sin fingir que el efecto principal de un DEM constante sea identificable en cada píxel. Compara términos lineales y términos no lineales, usando el histórico completo anterior al corte, climatología WRF, vecindario interpolado, hora, anomalías y cambios causales.

## Protocolo y reanudación

READY R7 validado por SHA-256. C0–C2 exclusivamente; siete días interiores para elegir escala/penalización y refit con todo el período anterior al corte. Mezcla regional-local de pesos fijos y diagnóstico por canal seleccionado exclusivamente en cortes anteriores; C0 es identidad en ese último diagnóstico. Períodos previamente explorados. Dos hilos de CPU, sin CUDA ni datos externos. No se evalúa C3.

Desde el worktree, usar el Python del proyecto original:

```bash
OMP_NUM_THREADS=2 ../Hackathon-IA-PUCP-2026/.venv/Scripts/python.exe -B -m train.agent_b.round7.run linear rbf clock clock-lags mos-linear mos-nonlinear regional-local channel-oof
OMP_NUM_THREADS=2 ../Hackathon-IA-PUCP-2026/.venv/Scripts/python.exe -B -m train.agent_b.round7.run --close
```

Las seis configuraciones entrenadas y las dos comparaciones que reutilizan sus predicciones se registran por separado al interpretar el presupuesto. No ejecutar dos escritores B en paralelo. La ejecución parcial se retoma usando los resúmenes auditados por corte. Los resultados y hashes están en `.private/experiments/round7/agent_b/` en el proyecto original, no en Git.

Se detuvo exclusivamente la corrida de Wind-Topo de B por instrucción del usuario; no se borró ningún artefacto. Su U/V cartesiano a 25 épocas dio 0.425582 agrupado en desarrollo frente a 0.476757 de R5; las variantes largas polar y física no completaron un resultado agrupado.

## Resultados verificados

| Configuración | C0–C2 | RH (%) | T (K) | U (m/s) | V (m/s) | s |
|---|---:|---:|---:|---:|---:|---:|
| `oof-0.25` (R4) | 0.475378 | 9.2713 | 0.8916 | 1.6196 | 1.4536 | — |
| KRR lineal | 0.453917 | 9.8198 | 0.9030 | 1.6983 | 1.5147 | 34.26 |
| KRR RBF | 0.433973 | 11.2382 | 0.9882 | 1.6869 | 1.4873 | 49.83 |
| KRR RBF×hora | 0.439208 | 11.2081 | 0.9859 | 1.6688 | 1.4625 | 48.20 |
| KRR RBF×hora, lags 2/1/0 h | 0.441530 | 11.1617 | 0.9835 | 1.6593 | 1.4568 | 41.68 |
| MOS lineal | 0.427503 | 10.2479 | 1.0097 | 1.7412 | 1.5587 | 51.29 |
| MOS no lineal | 0.432806 | 10.2851 | 1.0020 | 1.7155 | 1.5392 | 78.61 |
| Regional + local (50/50) | 0.461114 | 9.9382 | 0.9434 | 1.6333 | 1.4548 | 10.41 |
| `channel-oof` | 0.476710 | 9.2551 | 0.8916 | 1.6196 | 1.4438 | 9.71 |

El MOS no lineal mejora al MOS lineal pero no a R4. El control KRR lineal gana globalmente a los RBF directos; reloj y rezagos mejoran V dentro de los RBF. La media regional-local mejora a sus dos componentes aislados pero pierde frente a R4. Las mezclas fijas por canal al 25% se documentan como exploratorias (no se seleccionó con el corte objetivo); el KRR lineal aporta señal en los cuatro canales en el agregado y `clock-lags` reduce V en 0.01707 m/s en ese diagnóstico.

`channel-oof` aporta +0.001332 agrupado: empata en C0 y gana +0.003378/+0.000865 en C1/C2; 4/6 semanas mejoran. Selecciona 25% del KRR lineal para RH en C1, y 25% de `clock-lags` para V en C1/C2. T y U quedan intactos. El artefacto para A restaura RH/T a R4 por contrato, alcanza 0.476550 (+0.001172) y no llega al umbral +0.003. No promover para integración ni inferir validación independiente; guardar como especialista de V para investigación futura.

Pico de memoria residente del proceso ≈1,545 MiB; 445.70 s de reloj entre inicio y congelamiento, que incluyen reanudación, verificaciones e intervalos entre comandos. Se auditaron los 24 resultados por corte y se reconstruyeron los KRR lineal y `clock-lags` desde sus checkpoints sin refit. Los tiempos por candidato corresponden a la ejecución registrada; `clock-lags` reutilizó trabajo previo en una reanudación. Se verificaron el hash del protocolo, el hash del estado y los hashes de las tres predicciones del `DONE.json`; el manifiesto se conserva sin cambios para A.

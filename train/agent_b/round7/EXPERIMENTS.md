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

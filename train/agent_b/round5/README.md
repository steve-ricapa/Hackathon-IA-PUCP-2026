# Round 5 — ampliación del benchmark de desarrollo

Round 5 conserva el candidato de Round 4 y escribe sus propios artefactos privados en `.private/experiments/round5/agent_b/`. El benchmark no acepta C3. Usa el entorno del proyecto original y no requiere dependencias nuevas ni CUDA.

Ejecutar en orden `spectral`, `regime`, `regional`, `ensemble` y `audit`:

```bash
OMP_NUM_THREADS=3 ../Hackathon-IA-PUCP-2026/.venv/Scripts/python.exe -B -m train.agent_b.round5.loop spectral
```

Cambiar `spectral` por la etapa correspondiente. No ejecutar etapas simultáneamente sobre el mismo estado. Si una corrida se interrumpe, repetir la etapa: verifica las salidas completas, conserva las parciales y continúa. Al alcanzar el límite de la sesión no modificar su estado; abrir otra ronda con un límite nuevo y referenciar el manifiesto anterior. Para pruebas unitarias, ejecutar `python -B -m unittest train.agent_b.round5.test_round5` con el mismo intérprete.

## Evaluación

El filtro de promoción exige ventaja en score agrupado C0–C2, ningún corte peor y mejora en al menos tres de seis ventanas de siete días. Hay dos replays: seleccionar con cortes anteriores y evaluar en el siguiente; y seleccionar con los cortes anteriores más la semana uno, evaluar en la semana dos. Las predicciones de cada candidato están fijas durante la selección. Los replays son diagnósticos sobre períodos conocidos, no validaciones independientes: la cartera de configuraciones evolucionó con resultados de C0–C2.

`python -B -m train.agent_b.round5.diagnostics` resume el estado. La opción `--replay-fixed` reconstruye, si el ganador es una mezcla espectral fija, sus predicciones a partir del checkpoint espectral y del candidato preservado de Round 4, sin reajustar nada.

## Familias y referencias

- DCT y Haar de un nivel: ridge multisalida con bases espaciales compactas. La motivación espectral es de [Li et al. (2021)](https://arxiv.org/abs/2010.08895), pero el modelo implementado no es un FNO.
- Mezcla condicional: ridge sobre el desacuerdo de 13 miembros disponibles en los tres cortes; compuertas suaves basadas en GFS y particiones simples por latitud o relieve. En cada corte se ajusta solo con predicciones fuera de muestra de cortes anteriores.
- Análogos regionales y orografía: promedios suaves de vecinos en el historial de GFS, inspirados por [Delle Monache et al. (2013)](https://doi.org/10.1175/MWR-D-12-00281.1). Las salidas regionales se combinan con una partición de unidad. Se usan únicamente los datos oficiales.
- Ensambles por canal: selección y calibración con cortes anteriores; además, ablaciones fijas U/V elegidas tras observar el compromiso entre escalas espectrales y errores en RH/T durante desarrollo.

La guía metodológica para revisión y análisis fue [Kassis et al. (2026), *Scientific Agent Skills*](https://doi.org/10.48550/arXiv.2609.00065).

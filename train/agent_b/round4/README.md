# Round 4: especialistas U/V

El candidato es un postprocesador del ensemble congelado de Round 3, no un reemplazo. Ajusta ridge por píxel a residuos fuera de muestra de cortes anteriores, con predictores GFS causales. Aplica el 25% de la corrección a U y V; no toca RH ni T. Si no existen residuos anteriores, devuelve la referencia.

## Resultados negativos útiles

- Sustituir el 50% del viento de referencia no pasó el filtro de desarrollo en ninguna configuración. Las variantes directas de terreno, dinámica y RRR con lags largos tampoco pasaron al 25%.
- Con el mismo OOF al 25%, limitar la corrección a V, limitar las entradas a la hora, suavizar con 5×5 o subir la penalización no mejoró al finalista. No conviene repetir esas variantes sin cambiar la hipótesis.
- En EOF con kernel, 16 modos al 25% mejoraron los tres cortes de desarrollo; la reducción a 8 modos no funcionó. No se evaluó este segundo candidato en C3.

## Reproducción local

Ejecutar desde la raíz del worktree, junto al repositorio original. El código resuelve las rutas absolutas al dataset y los manifiestos privados de Round 3 en el original; no se necesita copiar datos. Los ajustes de esta ronda usan CPU.

Para verificar los resultados ya guardados:

```bash
../Hackathon-IA-PUCP-2026/.venv/Scripts/python.exe -B -m train.agent_b.round4.verify audit-final
```

En un worktree limpio **sin resultados previos** del mismo experimento, el orden es `run develop`, `verify freeze`, `run final`, `verify audit-final`, con `run` y `verify` como módulos de `train.agent_b.round4`. No sobrescribir `best_config.json` ni los artefactos de la corrida. El registro completo, las métricas y el índice SHA-256 quedan fuera de Git, en `.private/experiments/round4/agent_b/`.

Para A: conservar el ensemble sin corrección como valor por defecto. Antes de activar el postprocesador, confirmar en un bloque no explorado con un ensemble compatible con la calibración. No usar C3 para ajustar pesos, regiones ni regímenes.

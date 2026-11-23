# Reporte del producto · [Nombre del equipo]

> **Plantilla.** Copia este archivo a la raíz del repositorio de tu equipo y responde cada sección. Sé breve y concreto: 1 a 2 párrafos por pregunta, con números donde puedas. El jurado lo lee durante la preselección.

**Integrantes:** [Nombre 1], [Nombre 2], [Nombre 3], [Nombre 4]
**Producto:** [link al aplicativo o "se ejecuta en local, ver README"]
**Video pitch:** [link]

---

## 1. ¿Qué modelo o arquitectura usaron?

Describe tu modelo de downscaling: arquitectura (por ejemplo, CNN, U-Net, ConvLSTM, transformer), qué entradas usa (horas del GFS, relieve, variables extra creadas a partir de la data oficial), la función de pérdida y cómo validaste (qué corte temporal).

| | |
|---|---|
| Arquitectura | |
| Parámetros / tamaño de los pesos | |
| Score en el leaderboard público | |
| Score en tu validación local | |
| Tiempo de inferencia (24 horas de pronóstico) y hardware | |

## 2. ¿Para quién es y qué problema le resuelve?

¿Quién es tu usuario concreto (agricultor, guía de montaña, municipalidad…)? ¿Qué decisión/accion/etc toma hoy sin buena información y qué decide/toma/etc distinto con tu producto?

## 3. ¿Cuál es la propuesta de valor del producto?

En una o dos frases: qué gana el usuario al usar tu producto. Si puedes, cuantifícalo (tiempo, dinero, riesgo evitado, alcance).

## 4. ¿Por qué tu producto es diferente a algo usual?

¿Qué lo distingue de una app de clima común o de un mapa con el pronóstico? ¿Qué hace que el usuario lo prefiera?

## 5. ¿Cómo usa el producto la salida de tu modelo?

Explica el flujo de punta a punta: de dónde sale el GFS, cómo pasa por tu modelo y qué hace el producto con los mapas de 3 km. Si agregaste otras herramientas alrededor (LLM, traducción, alertas, otras fuentes de datos), di qué aporta cada una.

## 6. Limitaciones y próximos pasos

¿Dónde falla o es menos confiable tu modelo o tu producto (zonas, variables, horas)? ¿Que podrías agregarle al producto? ¿Qué harían con más tiempo?

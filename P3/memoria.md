---
title: Práctica 3
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
colorlinks: true
toc-depth: 2
toc-title: Índice
header-includes:
- \newcommand{\x}{\mathbf{x}}
- \newcommand{\w}{\mathbf{w}}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
---

\newpage

# Introducción: problema a resolver

## Descripción de los problemas

### *Optical Recognition of Handwritten Digits*

El primer dataset es un dataset de **clasificación**, que ya hemos utilizado en las prácticas anteriores.
Se trata de un conjunto de imágenes de dígitos manuscritos y debemos decir, a partir de su representación gráfica, de qué dígito se trata.

Cada dígito se ha escaneado con una resolución de 32x32, siendo cada píxel blanco o negro.
Posteriormente, para cada bloque 4x4 no solapado se ha calculado el número de píxeles negros que hay en ese bloque.
Esto nos da 64 atributos que tienen un valor entero entre 0 y 16.

Por último, la salida es un atributo categórico entre 0 y 9 que representa el dígito manuscrito.

Los datos han sido preprocesados de la manera anteriormente descrita.
Además, ya se han dividido en training y test.

### *Airfoil Self-Noise Data Set*

El segundo dataset es un dataset de **regresión**, con 5 características de entrada y una de salida.

El objetivo es intentar predecir el ruido que provoca un perfil de ala en un tunel de viento cuando se le expone con distintas velocidades y *ángulos de ataque*.
Los datos de entrada son

1. la frecuencia en hertzios,
2. el ángulo de ataaque en grados,
3. longitud del ala (*chord length*) en metros,
4. velocidad de corriente en metros por segundo y
5. espesor de desplazamiento (*suction side displacement thickness*) en metros.

El único dato de salida es el sonido generado (*scaled sound pressure level*) en decibelios.

Los datos no tienen ningún tipo de preprocesamiento.

## Visualización de los datos

# Preprocesado de los datos

# Selección de clase de hipótesis

# Conjuntos de training, validación y test

# Regularización

# Selección del modelo lineal

# Modelo final

# Métrica del ajuste

# Estimación del error real

# Discusión

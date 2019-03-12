---
title: Práctica 1
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
documentclass: scrartcl
---

# 1. Búsqueda iterativa de óptimos

## 1. Implementar el algoritmo de gradiente descendente
## 2. Encontrar el mínimo de $E$ usando gradiente descendente
### a) ¿Cuál es la expresión de $\grad E$?
### b) ¿Cuántas iteraciones tarda el algoritmo en obtener un valor inferior a $10^{-14}$?
### c) ¿En qué coordenadas se alcanza este valor?
## 3. Encontrar el mínimo de $f$ usando gradiente descendente
### a) Minimizar $f$ con gradiente descendente variando $\eta$
### b) Obtener el valor mínimo en función del punto de inicio

| $\mathbf{(x_0,y_0)}$ | $\mathbf{(x_{\operatorname{min}},y_{\operatorname{min}})}$ | $\mathbf{E(x_{\operatorname{min}},y_{\operatorname{min}})}$ | 
|----------------------|------------------------------------------------------------|-------------------------------------------------------------|  
| $(0.1, 0.1)$         |                                                            |                                                             |
| $(1, 1)$             |                                                            |                                                             |
| $(-0.5, -0.5)$       |                                                            |                                                             |
| $(-1, -1)$           |                                                            |                                                             |
## 4. ¿Cuál es la conclusión para encontrar el mínimo de una función arbitraria?


# 2. Regresión lineal
## 1. Estimación de un modelo de regresión lineal
### Implementación usando pseudoinversa
### Implementación usando SGD
### Pintar las soluciones obtenidas
### Valorar la bondad del resultado
## 2. Realizar el siguiente experimento
### a) Generar y mostrar muestra de entrenamiento
### b) Generar mapa de etiquetas usando $f$
### c) Ajustar un modelo de regresión lineal usando SGD
### d) Ejecutar el experimento 1000 veces
### e) Valoración

# Bonus

## 1. Repetir experimento haciendo uso del método de Newton
### Implementación del algoritmo
### Generar un gráfico de cómo desciende el valor de la función
### Extraer conclusiones y comparar

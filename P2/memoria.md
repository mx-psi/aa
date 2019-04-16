---
title: Práctica 2
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
toc-depth: 2
toc-title: Índice
header-includes:
- \newcommand{\x}{\mathbf{x}}
- \newcommand{\w}{\mathbf{w}}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
---

\newpage

# 1. Ejercicio sobre la complejidad de $\mathcal{H}$ y el ruido

> En este ejercicio debemos aprender la dificultad que introduce la aparición de ruido en las
> etiquetas a la hora de elegir la clase de funciones más adecuada. Haremos uso de tres funciones
> ya programadas:
>
> - `simula_unif(N, dim, rango)`, que calcula una lista de `N` vectores de dimensión `dim`. 
>    Cada vector contiene `dim` números aleatorios uniformes en el intervalo `rango`.
> - `simula_gaus(N, dim, sigma)`, que calcula una lista de `N` de vectores de dimensión `dim`,
>    donde cada posición del vector contiene un número aleatorio extraido de una distribucción 
>    Gaussiana de media 0 y varianza dada, para cada dimension, por la posición del vector `sigma`.
> - `simula_recta(intervalo)`, que simula de forma aleatoria los parámetros, $v = (a, b)$ de una
>    recta, $y = ax + b$, que corta al cuadrado $[-50, 50] \times [-50, 50]$.

## 1. Dibujar una gráfica con la nube de puntos de salida correspondiente.

### a) Considere $N = 50$, $\operatorname{dim} = 2$, $\operatorname{rango} = [-50, +50]$ con `simula_unif(N, dim, rango)`.

### b) Considere $N = 50$, $\operatorname{dim} = 2$ y $\sigma = [5, 7]$ con `simula_gaus(N, dim, sigma)`

## 2. Generar una muestra de puntos 2D con etiquetas según el lado al que queden de una recta

> Con ayuda de la función `simula_unif` generar una muestra de puntos 2D a los que vamos añadir una etiqueta 
> usando el signo de la función $f(x, y) = y - ax - b$, es decir el signo de la distancia de cada punto a la recta 
> simulada con `simula_recta`.

Nota: $N = 50$, intervalo $[-50,50]$.

### a) Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello. (Observe que todos los puntos están bien clasificados respecto de la recta)

### b) Modifique de forma aleatoria un 10 % etiquetas positivas y otro 10 % de negativas y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior. (Ahora hay puntos mal clasificados respecto de la recta)

## 3. Comparación de clasificadores

> Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
> 
> - $f (x, y) = (x - 10)^2 + (y - 20)^2 - 400$
> - $f (x, y) = 0.5(x + 10)^2 + (y - 20)^2 - 400$
> - $f (x, y) = 0.5(x - 10)^2 - (y + 20)^2 - 400$
> - $f (x, y) = y - 20x^2 - 5x + 3$
>
> Visualizar el etiquetado generado en 2b junto con cada una de las gráficas de cada una de
> las funciones. Comparar las formas de las regiones positivas y negativas de estas nuevas
> funciones con las obtenidas en el caso de la recta ¿Son estas funciones más complejas
> mejores clasificadores que la función lineal? ¿En que ganan a la función lineal? Explicar el
> razonamiento.

Mostrar el porcentaje de puntos que no clasificamos bien

# 2. Modelos lineales
## 1. Algoritmo Perceptron

> Implementar la función `ajusta_PLA(datos, label, max_iter, vini)`
> que calcula el hiperplano solución a un problema de clasificación binaria usando el algoritmo
> PLA. La entrada `datos` es una matriz donde cada item con su etiqueta está representado
> por una fila de la matriz, `label` el vector de etiquetas (cada etiqueta es un valor +1 o -1),
> `max_iter` es el número máximo de iteraciones permitidas y `vini` el valor inicial del vector.
> La función devuelve los coeficientes del hiperplano.

### a) Ejecutar el algoritmo PLA con los datos simulados en los apartados 2a de la sección.1.

> Inicializar el algoritmo con: a) el vector cero y, b) con vectores de números aleatorios
> en [0, 1] (10 veces). Anotar el número medio de iteraciones necesarias en ambos para
> converger. Valorar el resultado relacionando el punto de inicio con el número de
>iteraciones.

#### Inicializado con el vector cero

#### Inicializado con vectores de números aleatorios

#### Valoración del resultado

### b) Hacer lo mismo que antes usando ahora los datos del apartado 2b de la sección.1. ¿Observa algún comportamiento diferente? En caso afirmativo diga cual y las razones para que ello ocurra.

## 2. Regresión logística

> En este ejercicio crearemos nuestra propia función
> objetivo $f$ (una probabilidad en este caso) y nuestro conjunto de datos $\mathcal{D}$ 
> para ver cómo funciona regresión logística. Supondremos por simplicidad que $f$ es una probabilidad con
> valores $\{0,1\}$ y por tanto que la etiqueta $y$ es una función determinista de $\x$.
> Consideremos $d = 2$ para que los datos sean visualizables, y sea $\mathcal{X} = [0, 2] \times [0, 2]$
> con probabilidad uniforme de elegir cada $\x \in \mathcal{X}$ . 
> Elegir una línea en el plano que pase por $\mathcal{X}$ como la frontera entre $f(\x) = 1$ 
> (donde $y$ toma valores $+1$) 
> y $f(\x) = 0$ (donde y toma valores $-1$), para ello seleccionar dos puntos aleatorios del plano 
> y calcular la línea que pasa por ambos. 
> Seleccionar $N = 100$ puntos aleatorios $\{\x_n\}$ de $\mathcal{X}$ y evaluar las respuestas
> $\{y_n\}$ de todos ellos respecto de la frontera elegida.

### a)  Implementar Regresión Logística (RL) con Gradiente Descendente Estocástico (SGD) 

>  Implementar Regresión Logística(RL) con Gradiente Descendente Estocástico (SGD)
>  bajo las siguientes condiciones:
> 
> - Inicializar el vector de pesos con valores 0.
> - Parar el algoritmo cuando $\lVert \w^{(t-1)} - \w^{(t)}\rVert < 0.01$, donde $\w(t)$ denota el vector
>   de pesos al final de la época $t$. Una época es un pase completo a través de los $N$
>   datos.
> - Aplicar una permutación aleatoria, $1, 2, \dots, N$ , en el orden de los datos antes de
>   usarlos en cada época del algoritmo.
> - Usar una tasa de aprendizaje de $\eta = 0.01$

Las etiquetas siguen siendo -1 y +1
Utilizar SGD con batch_size de 1
Si el batch es grande podríamos tener problemas de convergencia

### b) Usar la muestra de datos etiquetada para encontrar nuestra solución $g$ y estimar $E_{\operatorname{out}}$ usando para ello un número suficientemente grande de nuevas muestras ($>999$).


# 3. Bonus

## Clasificación de dígitos

> (1.5 puntos) Clasificación de Dígitos. Considerar el conjunto de datos de los dígitos manuscritos 
> y seleccionar las muestras de los dígitos 4 y 8. Usar los ficheros de entrenamiento (training)
> y test que se proporcionan. Extraer las características de **intensidad promedio** y **simetría** en
> la manera que se indicó en el ejercicio 3 del trabajo 1.

### 1. Plantear un problema de clasificación binaria que considere el conjunto de entrenamiento como datos de entrada para aprender la función $g$.

### 2. Usar un modelo de Regresión Lineal y aplicar PLA-Pocket como mejora. Responder a las siguientes cuestiones.

#### a) Generar gráficos separados (en color) de los datos de entrenamiento y test junto con la función estimada.

#### b) Calcular $E_{\operatorname{in}}$ y $E_{\operatorname{test}}$ (error sobre los datos de test).

#### c) Obtener cotas sobre el verdadero valor de $E_{\operatorname{out}}$ . Pueden calcularse dos cotas una basada en $E_{\operatorname{in}}$ y otra basada en $E_{\operatorname{test}}$ . Usar una tolerancia $\delta= 0.05$. ¿Que cota es mejor?



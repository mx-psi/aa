---
title: Cuestionario 3
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
toc-depth: 1
toc-title: Índice
header-includes:
- \usepackage{stmaryrd}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
- \usepackage{algorithm}
- \usepackage{algpseudocode}
---

\newpage

# Pregunta 1

>  ¿Podría considerarse Bagging como una técnica para estimar el error de predicción de un
> modelo de aprendizaje?. Diga si o no con argumentos. En caso afirmativo compárela con
> validación cruzada.

*Bagging* es una técnica para reducir la varianza de un modelo utilizando muchos conjuntos de datos de entrenamiento distintos generados con reemplazamiento del conjunto de datos original y promediando sus resultados (o tomando el voto mayoritario en el caso de clasificación).

Utiliza la técnica de **bootstraping**, que puede considerarse una ténica para estimar el error de predicción

# Pregunta 2

>  Considere que dispone de un conjunto de datos linealmente separable. Recuerde que una
> vez establecido un orden sobre los datos, el algoritmo perceptron encuentra un hiperplano
> separador interando sobre los datos y adaptando los pesos de acuerdo al algoritmo

\begin{algorithm}
\caption{Perceptron}
\begin{algorithmic}[1]
\State \textbf{Entradas:} $(\mathbf{x}_i,y_i), i = 1,\dots,n, \mathbf{w} = 0, k = 0$
\Repeat
\State $k \gets (k+1) \mod n$
\If{$\operatorname{sign}(y_i) \neq \operatorname{sign}(\mathbf{w}^T \mathbf{x}_i)$}
\State $\mathbf{w} \gets \mathbf{w} + y_i\mathbf{x}_i$
\EndIf
\Until{todos los puntos bien clasificados}
\end{algorithmic}
\end{algorithm}

> Modificar este pseudo-código para adaptarlo a un algoritmo simple de SVM, considerando
> que en cada iteración adaptamos los pesos de acuerdo al caso peor clasificado de toda la
> muestra. Justificar adecuadamente/matematicamente el resultado, mostrando que al final
> del entrenamiento solo estaremos adaptando los vectores soporte.

En el caso separable

La modificación está en la condición para la actualización.

\begin{algorithm}
\caption{SVM}
\begin{algorithmic}[1]
\State \textbf{Entradas:} $(\mathbf{x}_i,y_i), i = 1,\dots,n, \mathbf{w} = 0, k = 0$
\Repeat
\State $k \gets (k+1) \mod n$
\If{$y_i (\mathbf{w}^T \mathbf{x}_i) < 1$}
\State $\mathbf{w} \gets \mathbf{w} + y_i\mathbf{x}_i$
\EndIf
\Until{todos los puntos bien clasificados}
\end{algorithmic}
\end{algorithm}

\newpage

# Pregunta 3

> Considerar un modelo SVM y los siguientes datos de entrenamiento: 
>
> Clase-1
> : $\{(1,1),(2,2),(2,0)\}$,
>
> Clase-2
> : $\{(0,0),(1,0),(0,1)\}$

## a)  Dibujar los puntos y construir por inspección el vector de pesos para el hiperplano óptimo y el margen óptimo.

La siguiente figura uestra en azul los puntos de la clase 1 y en verde los puntos de la clase 2, junto con el margen óptimo en coloreado en amarillo:
![Representación de los puntos y el modelo SVM](ej2.png)

Como vemos el hiperplano clasifica correctamente los puntos en función del lado al que quedan y además su distancia mínima a un punto es la máxima posible (representada por el margen marcado en amarillo).

## b) ¿Cuáles son los vectores soporte?

Los vectores soporte son aquellos que definen el hiperplano, esto es, aquellos que tienen la mínima distancia con el hiperplano.

Como muestra el diagrama, los vectores soporte del hiperplano son los puntos $(0,1),(1,0),(1,1),(2,2)$.

## c)  Construir la solución en el espacio dual. Comparar la solución con la del apartado (a)

# Pregunta 4

## ¿Cuál es el criterio de optimalidad en la construcción de un árbol?

## Analice un clasificador en árbol en términos de sesgo y varianza. 

## ¿Que estrategia de mejora propondría?

# Pregunta 5

> ¿Cómo influye la dimensión del vector de entrada en los modelos

## SVM
## RF
## Boosting

## NN


# Pregunta 6

> El método de Boosting representa una forma alternativa en la búsqueda del mejor clasificador
> respecto del enfoque tradicional implementado por los algoritmos PLA, SVM, NN, etc. 

## a) Identifique de forma clara y concisa las novedades del enfoque
## b) Diga las razones profundas por las que la técnica funciona produciendo buenos ajustes
## c) Identifique sus principales debilidades
## d) ¿Cuál es su capacidad de generalización comparado con SVM?

# Pregunta 7

> Discuta pros y contras de los clasificadores SVM y Random Forest (RF). Considera que
> SVM por su construcción a través de un problema de optimización debería ser un mejor
> clasificador que RF. Justificar las respuestas.

# Pregunta 8

## ¿Cuál es a su criterio lo que permite a clasificadores como Random Forest basados en un conjunto de clasificadores simples aprender de forma más eficiente? 
## ¿Cuales son las mejoras que introduce frente a los clasificadores simples? 
## ¿Es Random Forest óptimo en algún sentido?

# Pregunta 9

>  En un experimento para determinar la distribución del tamaño de los peces en un lago, se
> decide echar una red para capturar una muestra representativa. Así se hace y se obtiene
> una muestra suficientemente grande de la que se pueden obtener conclusiones estadísticas
> sobre los peces del lago. Se obtiene la distribución de peces por tamaño y se entregan las
> conclusiones. Discuta si las conclusiones obtenidas servirán para el objetivo que se persigue
> e identifique si hay algo que lo impida.

# Pregunta 10

> Identifique que pasos daría y en que orden para conseguir con el menor esfuerzo posible un
> buen modelo de red neuronal a partir una muestra de datos. Justifique los pasos propuestos,
> el orden de los mismos y argumente que son adecuados para conseguir un buen óptimo.
> Considere que tiene suficientes datos tanto para el ajuste como para el test.


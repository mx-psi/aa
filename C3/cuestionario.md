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

*Bagging* es una técnica para reducir la varianza de un modelo.
Dado un conjunto de entrenamiento $X$ de tamaño $N$, muestrea conjuntos de datos de entrenamiento de tamaño $N$ distintos generados con reemplazamiento del conjunto de datos original (com probable repetición) (esta etapa se conoce como *bootstrapping*) y une sus resultados, tomando el voto mayoritario en el caso de clasificación o promediando en el caso de regresión para obtener el resultado final.

**Sí** podemos utilizar *bootstrapping* para estimar el error de predicción: en concreto estamos viendo cómo varía el error del predictor en función de los datos muestreados del conjunto de datos de entrenamiento completo.
De media, utilizamos aproximadamente dos tercios de los datos en cada clasificador con repeticiones.

En la validación cruzada, podemos establecer una partición en $K$ conjuntos de tamaño similar, entrenamos con $K-1$ de ellos y estimamos el error con el restante.
En este caso no hay repetición de los datos entre los que usamos para entrenar.
Con esta técnica también podemos estimar el error de predicción y reducir la varianza.
Ambas técnicas serán más similares cuando $K = 3$, pero difieren en que validación cruzada no hay posibilidad de repetición.

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

> ¿Cuál es el criterio de optimalidad en la construcción de un árbol?

En la construcción de un árbol el criterio es doble: buscamos un árbol que sea simple (en el sentido de que su descripción tenga menor longitud) y que además ajuste correctamente los datos (utilizando algún criterio de selección de características como la entropía o la ganancia de información).
Esto es, buscamos un balance entre la complejidad del árbol y la bondad del ajuste.
En la práctica, la estrategia óptima para ajustar un árbol es crear el árbol más grande (hasta que el número de items en las hojas esté por debajo de un cierto umbral) y después eliminar en función de su coste-complejidad, que mide este criterio.

> Analice un clasificador en árbol en términos de sesgo y varianza.

Los árboles son una clase de hipótesis de dimensión VC infinita y por tanto con **bajo sesgo**: son un modelo no lineal que puede ajustarse con precisión a los datos de entrenamiento.

Esto, sin embargo, provoca que también tengan **alta varianza**, esto es, con datos de la misma distribución podemos obtener resultados muy diferentes. Son por tanto propensos al sobreajuste si no se utilizan estrategias que limiten este problema.

> ¿Que estrategia de mejora propondría?

Una posible estrategia de mejora es utilizar una técnica de reducción de la varianza como *bagging*, descrita en la [Pregunta 1], que combina varios árboles en un único clasificador que da como resultado la media de los resultados de los árboles.

También podemos seleccionar una cantidad limitada de características para limitar la correlación entre los distints árboles que surge de tener características muy relevantes; este es el principio detrás del modelo de Random Forest. 

Ambas técnicas nos permiten seguir teniendo este ajuste no lineal reduciendo la varianza.

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

El método de Boosting se basa en la combinación de un conjunto de estimadores débiles que sólo son un poco mejores que el azar mediante su suma ponderada.

Estos estimadores débiles se van creando sucesivamente utilizando el mismo algoritmo, pero dando distintos pesos a las características, en función de aquellas que sean más relevantes para reducir el error actual.

La salida final será la media ponderada de la respuesta que da cada estimador débil.
Esta ponderación se calcula para reducir el error general.

Su enfoque alternativo con respecto al resto de algoritmos mencionados en el enunciado es el uso de muchos estimadores débiles y su combinación para obtener un buen estimador en lugar de utilizar un estimador más complejo como puede ser un modelo lineal general.

## b) Diga las razones profundas por las que la técnica funciona produciendo buenos ajustes

Esta técnica produce 

## c) Identifique sus principales debilidades

Su principal debilidad es que la función de error utilizada, el error exponencial, 

## d) ¿Cuál es su capacidad de generalización comparado con SVM?

# Pregunta 7

> Discuta pros y contras de los clasificadores SVM y Random Forest (RF). ¿Considera que
> SVM por su construcción a través de un problema de optimización debería ser un mejor
> clasificador que RF?. Justificar las respuestas.

Los clasificadores Random Forest tienen un menor sesgo por tener una clase de hipótesis más extensa: la dimensión VC de la clase de hipótesis generada por RF es infinita.
En cambio, en el caso de los clasificadores SVM (sin usar el *kernel trick*) la clase de hipótesis fijada tiene dimensión finita.
Esto supone sin embargo un inconveniente para Random Forest: puede tener mayor varianza y sufre la necesidad de usar un paradigma de aprendizaje que no es tan simple como la reducción del error empírico de ERM.

Además, el uso de Random Forest es además ventajoso para más tipos de datos:
para la clasificación multiclase, Random Forest tiene la ventaja de que no necesita ninguna preparación especial, mientras que SVM necesita adaptar a un problema *One v. Rest* u otra estrategia, y Random Forest puede usarse con variables categóricas sin ninguna preparación especial.

SVM no es necesariamente mejor que RF: por el teorema de *No Free Lunch* qué algoritmo de aprendizaje es mejor dependerá del dominio del problema con el que estemos tratando, y por tanto no podemos decir que uno es mejor que otro en todos los casos.

# Pregunta 8

> ¿Cuál es a su criterio lo que permite a clasificadores como Random Forest basados en un conjunto de clasificadores simples aprender de forma más eficiente? 
> ¿Cuales son las mejoras que introduce frente a los clasificadores simples? 
> ¿Es Random Forest óptimo en algún sentido?

Lo que permite a esta clase de clasificadores aprender de forma más eficiente es la combinación de clasificadores simples no correlados que tienen bajo sesgo para la reducción de la varianza.

Las dos mejoras principales de Random Forest son que:

1. reduce la varianza de los clasificadores simples mediante el promedio o voto mayoritario de sus componentes, 
   mediante la técnica de bagging descrita en la [pregunta 1] y
2. reduce la correlación entre estos clasificadores mediante la selección aleatoria de características.
   Esta reducción de la correlación se da porque si hay características muy decisivas para la clasificación es 
   posible que los distintos árboles estén muy correlacionados.
   
Aunque en la práctica puede ser un tipo de predictor muy útil, ningún clasificador puede ser óptimo por el teorema de *No Free Lunch*, por lo que no podemos decir que Random Forest lo sea.

# Pregunta 9

>  En un experimento para determinar la distribución del tamaño de los peces en un lago, se
> decide echar una red para capturar una muestra representativa. Así se hace y se obtiene
> una muestra suficientemente grande de la que se pueden obtener conclusiones estadísticas
> sobre los peces del lago. Se obtiene la distribución de peces por tamaño y se entregan las
> conclusiones. Discuta si las conclusiones obtenidas servirán para el objetivo que se persigue
> e identifique si hay algo que lo impida.

Para que este método nos sirva para obtener conclusiones sobre la distribución de tamaños de los peces las muestras deben ser independientes idénticamente distribuidas, proviniendo de la distribución que queremos estudiar.
Además, el número de estas muestras debe ser suficientemente grande como para que, bajo las condiciones anteriores,
podamos asegurar con suficiente probabilidad que estamos representando la distribución de forma adecuada.

Si bien la condición del tamaño de la muestra sí puede cumplirse según el enunciado,
el método indicado puede inducir a un sesgo en el muestreo realizado que implique que las muestras no provengan de la distribución que queremos medir o que no sean independientes. 
En concreto, podemos identificar al menos dos fuentes de sesgo:

1. Los peces de cierto tamaño pueden ser más propensos de ser capturados por nuestra red y
2. los peces del mismo tamaño pueden ser más propensos a estar en la misma localización.

Por tanto, las conclusiones obtenidas podrían **no** servir para el objetivo que se persigue, ya que no estar bajo las condiciones de i.i.d. impediría que estas conclusiones representara la distribución que queríamos medir.

# Pregunta 10

> Identifique que pasos daría y en que orden para conseguir con el menor esfuerzo posible un
> buen modelo de red neuronal a partir una muestra de datos. Justifique los pasos propuestos,
> el orden de los mismos y argumente que son adecuados para conseguir un buen óptimo.
> Considere que tiene suficientes datos tanto para el ajuste como para el test.


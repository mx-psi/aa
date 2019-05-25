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

\newpage

> Modificar este pseudo-código para adaptarlo a un algoritmo simple de SVM, considerando
> que en cada iteración adaptamos los pesos de acuerdo al caso peor clasificado de toda la
> muestra. Justificar adecuadamente/matematicamente el resultado, mostrando que al final
> del entrenamiento solo estaremos adaptando los vectores soporte.

La modificación que nos indica el enunciado está en la condición para la actualización.
En cada paso escogemos el índice del vector peor clasificado (esto es, el $k$ que tenga 
el mínimo valor de $y_k \mathbf{w}^T \mathbf{x}_k$) y actualizamos el vector de pesos en 
función de este.

El algoritmo quedaría
\begin{algorithm}
\caption{SVM}
\begin{algorithmic}[1]
\State \textbf{Entradas:} $(\mathbf{x}_i,y_i), i = 1,\dots,n, \mathbf{w} = 0, k = 0$
\Repeat
\State $i \gets \operatorname{arg} \min_k y_k \mathbf{w}^T \mathbf{x}_k$
\State $\mathbf{w} \gets \mathbf{w} + y_i\mathbf{x}_i$
\Until{todos los puntos bien clasificados}
\end{algorithmic}
\end{algorithm}

La condición de tener menor $y_k \mathbf{w}^T \mathbf{x}_k$ nos indica qué vector está peor clasificado,
ya que la distancia al hiperplano dado por $\mathbf{w}$ es esta cantidad dividida por el módulo de $\mathbf{w}$.

A partir de un número suficientemente grande de iteraciones sólo consideraremos los vectores soporte;
sólo consideraremos aquellos vectores a menor distancia del hiperplano e ignoraremos el resto.
Estos vectores a menor distancia son los vectores soporte.


# Pregunta 3

> Considerar un modelo SVM y los siguientes datos de entrenamiento: 
>
> Clase-1
> : $\{(1,1),(2,2),(2,0)\}$,
>
> Clase-2
> : $\{(0,0),(1,0),(0,1)\}$

## a)  Dibujar los puntos y construir por inspección el vector de pesos para el hiperplano óptimo y el margen óptimo.

Por inspección, el hiperplano óptimo será la recta con ecuación
$$x + y = -\frac{3}{2}.$$

La siguiente figura uestra en azul los puntos de la clase 1 y en verde los puntos de la clase 2, junto con el hiperplano y el margen óptimo en coloreado en amarillo:
![Representación de los puntos y el modelo SVM](ej2.png)

Como vemos el hiperplano clasifica correctamente los puntos en función del lado al que quedan y además su distancia mínima a un punto es la máxima posible (representada por el margen marcado en amarillo).

## b) ¿Cuáles son los vectores soporte?

Los vectores soporte son aquellos que definen el hiperplano, esto es, aquellos que tienen la mínima distancia con el hiperplano.

Como muestra el diagrama, los vectores soporte del hiperplano son los puntos $(0,1),(1,0),(1,1),(2,2)$.

## c)  Construir la solución en el espacio dual. Comparar la solución con la del apartado (a)

Para construir la solución en esl espacio dual he utilizado el paquete `qpsolvers`.
Nuestro objetivo es calcular los parámetros $\alpha$ que sean solución de un sistema de ecuaciones, la combinación lineal de las etiquetas se anule, $\sum \alpha_ny_n = 0$ y sean todos no negativos.

Los datos vienen dados por
```python
x = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
y = np.array([1, 1, 1, -1, -1, -1])
```

Construimos la matriz cuadrática, que vendrá dada por el producto matricial $$\mathtt{Q}_\mathtt{P} = \mathtt{x} \cdot \mathtt{y} \cdot (\mathtt{x} \cdot \mathtt{y})^T.$$ Esta matriz no es definida positiva (consta de una columna de ceros), por lo que no podemos utilizarla en el programa directamente.

Utilizamos una sencilla propiedad: la suma de una matriz definida positiva y una matriz semidefinida positiva es definida positiva.
En particular sabemos que $\mathtt{Q}_\mathtt{P}$ es semidefinida positiva, luego sólo tenemos que sumarle una matriz definida positiva que no perturbe demasiado los valores. Optamos por sumar la matriz identidad multiplicada por una pequeña constante.

Por tanto, podemos codificar las restricciones en el programa como sigue:
```python
z = y[:, np.newaxis] * x
Q_P = z.dot(z.T).astype(np.float) + 1e-8 * identity
identity = np.eye(N)

alpha = solve_qp(Q_P, -np.ones((N, )), 
             -identity, np.zeros((N, )), 
             y, 0)
```
donde los primeros dos argumentos definen el sistema de ecuaciones, los dos segundos la no negatividad de los $\alpha_i$ y los últimos que $\sum \alpha_ny_n = 0$.
Salvo escalado, si lo hacemos sin coordenadas homogéneas el resultado es $(1,1)$.

Recuperamos también el sesgo restando algún vector soporte, por ejemplo el primero:
```python
b = y[0] - w_dual.dot(x[0])
```
lo que nos da, salvo pequeño error de redondeo, la misma solución que el problema original, esto es
$$x + y = -\frac{3}{2}.$$



\newpage

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

> ¿Cómo influye la dimensión del vector de entrada en los [siguientes] modelos?

SVM
: Las máquinas de vectores de soporte son un buen modelo incluso cuando el espacio de los vectores de entrada tiene alta dimensionalidad, ya que conocemos que la dimensión VC del modelo es uno más que el mínimo entre la bola en la que podamos introducir los vectores de entrada (es decir, en su máxima norma) y la dimensión del espacio.
Por tanto, este algoritmo no depende necesariamente en la dimensión de entrada de cara a la bondad de su ajuste.

RF
: Como el número de características que seleccionamos para la generación de cada árbol suele ser la raíz del número total de características, esto hace que haya cierta dependencia de la dimensión del vector de entrada, que puede llevar a cierto sobreajuste cuando la dimensión del vector de entrada es muy muy alta.
No obstante, el promedio entre distintos árboles reduce este sobreajuste en cierta medida.

Boosting
: En el caso de boosting su dependencia de la dimensión del vector de entrada estará en la forma de los clasificadores débiles. En el caso de que los clasificadores sean muy complejos es posible que la dimensión del vector de entrada afecte en gran medida a la capacidad de afrontar el problema de un modelo que utilice boosting.

NN
: La dimensión VC de las redes neuronales depende quasilinealmente del número de dimensiones del espacio de entrada,
  aunque esto varía dependiendo del modelo concreto de red neuronal que utilicemos.
  Esto supone que las redes neuronales pueden tener problemas con un espacio de entrada de alta dimensionalidad si 
  no utilizamos métodos de regularización y validación cruzada.

# Pregunta 6

> El método de Boosting representa una forma alternativa en la búsqueda del mejor clasificador
> respecto del enfoque tradicional implementado por los algoritmos PLA, SVM, NN, etc. 

> a) Identifique de forma clara y concisa las novedades del enfoque

El método de Boosting se basa en la combinación de un conjunto de estimadores débiles que sólo son un poco mejores que el azar mediante su suma ponderada.

Estos estimadores débiles se van creando sucesivamente utilizando el mismo algoritmo, pero dando distintos pesos a las características, en función de aquellas que sean más relevantes para reducir el error de clasificación que se tiene en ese momento.

La salida final será la media ponderada de la respuesta que da cada estimador débil.
Esta ponderación se calcula para reducir el error general.

Su enfoque alternativo con respecto al resto de algoritmos mencionados en el enunciado es el uso de muchos estimadores débiles y su combinación para obtener un buen estimador en lugar de utilizar un estimador más complejo como puede ser un modelo lineal general.

> b) Diga las razones profundas por las que la técnica funciona produciendo buenos ajustes.

Su funcionamiento se debe a dos razones principales:

1. es capaz de ponderar las predicciones de varios predictores muy débiles en función de su importancia en la reducción del error y
2. es capaz de seleccionar y ponderar las características disponibles, seleccionando las más relevantes en cada momento.

Estas dos características son las que lo hacen una técnica útil en la práctica, y las que la hacen una técnica capaz de reducir el error en test incluso cuando el de training ya ha sido minimizado.

> c) Identifique sus principales debilidades

Sus principales debilidades están en el ruido de los datos y en el rendimiento de los clasificadores que se suman.

Boosting y en particular AdaBoost tiene problemas con el ruido uniforme y con los outliers, que pueden llevar a un mal rendimiento.

Además, si los clasificadores son demasiado complejos puede darse el sobreajuste, y si son demasiado simples podemos tener problemas a la hora de ajustar correctamente los datos.

>  d) ¿Cuál es su capacidad de generalización comparado con SVM?

Tanto SVM como las técnicas de boosting son aplicables para conjuntos de datos muy generales, pero en ambos tenemos que escoger algunos hiperparámetros.

En el caso de SVM tenemos que elegir el kernel a utilizar, mientras que en el caso de las técnicas de boosting la elección está en el algoritmo que elige los clasificadores débiles.

En general, boosting puede ser aplicable en más situaciones ya que la resolución de SVM mediante programación cuadrática puede ser computacionalmente muy intensiva cuando la dimensionalidad del espacio de los datos sea muy alta.


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

\newpage

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

\newpage

# Pregunta 10
  
> Identifique que pasos daría y en que orden para conseguir con el menor esfuerzo posible un
> buen modelo de red neuronal a partir una muestra de datos. Justifique los pasos propuestos,
> el orden de los mismos y argumente que son adecuados para conseguir un buen óptimo.
> Considere que tiene suficientes datos tanto para el ajuste como para el test.

En primer lugar, decidiría la estructura del modelo de red neuronal utilizada.
Las capas de entrada y salida tendrían un número de nodos igual a la dimensión de entrada y salida.
Estos datos quedan fijados por el dominio del problema.
Dejaría el número de capas ocultas como parámetro a elegir.

Inicializaría los pesos de todas las capas como indica la teoría, esto es, con valores pequeños y aleatorios que sigan una distribución gaussiana de media 0 y varianza acotada por el máximo de las normas de los datos de entrenamiento.
Esto promovería el movimiento hacia un mínimo local pero evitaría que en el algoritmo de *backpropagation* hubiera problemas con la propagación del error del gradiente por ser los pesos muy grandes.

El criterio de finalización sería una combinación de un máximo número de iteraciones y una condición sobre error en training suficientemente pequeño.
Si el número de iteraciones es bajo esto puede hacer que exploremos un área más pequeña del conjunto de hipótesis, lo que mejoraría la capacidad de generalización del modelo.

Por último, decidiría el número de capas ocultas mediante un proceso de validación cruzada en el que compararía los distintos posibles modelos. El número de capas ocultas variaría entre 1 y un número no muy alto para evitar el sobreajuste.
Este entrenamiento tendría algún tipo de regularización como el *weight-decay* para evitar el sobreajuste.

Estos pasos serían probablemente adecuados para obtener un buen óptimo, ya que así conseguiríamos con mayor probabilidad que el proceso de minimización del error llegara a un óptimo local que produjera sobreajuste y escogeríamos sin esfuerzo una arquitectura adecuada.


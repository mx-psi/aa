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
bibliography: citas.bib
biblio-style: apalike
link-citations: true
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
2. el ángulo de ataque en grados,
3. longitud del ala (*chord length*) en metros,
4. velocidad de corriente en metros por segundo y
5. espesor de desplazamiento (*suction side displacement thickness*) en metros.

El único dato de salida es el sonido generado (*scaled sound pressure level*) en decibelios.

Los datos no tienen ningún tipo de preprocesamiento.

\newpage

## Obtención de los datos

Los datos de cada problema se guardan en su carpeta dentro de la carpeta `datos`: 
en las subcarpetas `optdigits` y `airfoil` para los problemas de clasificación y regresión respectivamente.

He utilizado la versión preprocesada y dividida en training y test en el caso del problema de clasificación de dígitos.
He conservado los nombres de los ficheros originales en todos los casos (`optdigits.tra` y `optdigits.tes` para clasificación y `airfoil_self_noise.dat` para regresión).

Para crear fácilmente la carpeta de `datos` con la estructura que he utilizado en un sistema Unix pueden ejecutarse los siguientes comandos:

```sh
mkdir -p datos
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra \
 -O datos/optdigits.tra
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes \
 -O datos/optdigits.tes

mkdir -p datos/airfoil
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat \
 -O datos/airfoil_self_noise.dat
```

Alternativamente, pueden modificarse las constantes `DIGITS_TRA`, `DIGITS_TEST` y `AIRFOIL` que aparecen al comienzo del archivo para modifcar el lugar del que se obtienen los datos.

Para la lectura de datos he utilizado la función `loadtxt`, que lee los datos de un fichero en formato csv.
Como los separadores entre los distintos datos son diferentes en un dataset y otro debemos especificar este dato también. 
Separamos por último la última columna (la columna a predecir) del resto:

```python
def lee_datos(filename, delimiter):
  data = np.loadtxt(filename, delimiter=delimiter)
  return data[:, :-1], data[:, -1]
```

## Visualización de los datos

Antes de pasar al preprocesado, procedemos a una visualización de los datos para observar su estructura.
Utilizamos técnicas distintas para cada problema, debido a su diferente naturaleza.

### Clasificación: `optdigits`

**Nota**: La generación de la visualización de este apartado lleva aproximadamente un minuto en mi ordenador, debido a que es una técnica costosa computacionalmente, por lo que he añadido al script del trabajo una pregunta para comprobar si se desea o no hacer la generación de esta imagen.

****

En el caso del dataset de dígitos manuscritos, nos hallamos ante datos en un espacio de alta dimensionalidad, incluso después del preprocesado inicial con el que nos proveen los autores.

Para este tipo de datasets, puede ser útil el uso de una técnica de visualización de variedades.
Me he decidido por utilizar la técnica t-SNE, que nos permite visualizar en dos dimensiones un conjunto de alta dimensionalidad conservando parcialmente su estructura, a escalas pequeñas y grandes [@vanderMaatenVisualizingDatausing2008].

Para obtener la proyección a dos dimensiones podemos utilizar el objeto `TSNE` de la biblioteca `sklearn.manifold`:
```python
X_new = TSNE(n_components=2).fit_transform(digitos_tra_x)
```

A continuación he definido una función auxiliar `visualizar_clasif` para visualizar conjuntos de datos de problemas de clasificación en dos dimensiones. La función realiza un scatter plot (de forma idéntica a la función scatter que presenté en prácticas anteriores) y pinta además etiquetas de los puntos en el centroide de cada conjunto de puntos con la misma etiqueta. Si `x` e `y` son los datos y las etiquetas respectivamente, este etiquetado puede hacerse de la siguiente forma:
```python
labels = np.unique(y)
for label in labels:
  centroid = np.mean(x[y == label], axis=0)
  ax.annotate(int(label),
              centroid,
              size=14,
              weight="bold",
              color="white",
              backgroundcolor="black")
```

El resultado puede verse en la figura 1.

![Visualización del dataset `optdigits` en dos dimensiones usando t-SNE](img/tSNEdigitos.pdf)


Como puede apreciarse en la figura, los distintos números se agrupan de forma bastante diferenciada unos de otros en la proyección obtenida por t-SNE.
Debido a las distorsiones que se sufren necesariamente al reducir la dimensionalidad, las distancias en la figura y los tamaños de los clusters obtenidos no tienen por qué corresponderse con la distribución real de los puntos en el espacio original.[@WattenbergHowUsetSNE2016]

No obstante, podemos ver que los conjuntos de imágenes que parecen más difíciles de diferenciar son los correspondientes a los dígitos 1,8 y 9, y que los dígitos 9 parecen agruparse en dos estilos diferentes.

### Regresión: `airfoil`

En el caso de regresión, dado el reducido número de variables podemos proceder a observar la relación de la variable que queremos predecir (el ruido producido) con cada una de las variables de entrada de las que disponemos.

Para producir las imágenes hacemos un *scatter plot* de cada variable con la variable a predecir.
Por motivos de espacio lo hacemos en dos plots separados.
Discutimos brevemente el código de uno de ellos (ambos son idénticos salvo por el tamaño).

`airfoil_titles` es una lista de nombres de las variables de entrada y `airfoil_indep` el nombre de la variable de salida.
Hacemos un plot con varios subplots que comparten el eje y, indicando su etiqueta.
Además, usamos la función `add_common_ylabel` para añadir el eje de las y compartido:

```python
fig, axs = plt.subplots(1, 2, sharey=True, figsize=[11.0, 4.8])
add_common_ylabel(fig, airfoil_indep)
for i in [0, 1]:
  axs[i].scatter(airfoil_tra_x[:, i],
                 airfoil_tra_y,
                 alpha=0.5,
                 marker="v",
                 c="#687aed")
  axs[i].set(xlabel=airfoil_titles[i])
plt.title("Dependencia del ruido respecto de distintas variables (1)")
plt.show()
```

El resultado puede verse en las siguientes imágenes.

![](img/airfoil_1.pdf)
![](img/airfoil_2.pdf)

Como podemos ver en las imágenes, en general con respecto de una sola variable el ruido tiene una gran variabilidad incluso para un valor fijado de la variable de entrada.
Podemos apreciar una débil relación negativa entre la frecuencia y el espesor de desplazamiento y el sonido generado.

Además, podemos apreciar que las variables de longitud de ala y velocidad de corriente toman sólo 6 y 4 valores respectivamente.

Como conclusiones podemos deducir que algunas de las variables aportan mucha menos información que el resto cuando se presentan por separado y que los datos tienen potencialmente mucha variabilidad incluso cuando fijamos algunas de las variables.

\newpage


# Preprocesado de los datos

## Uso de Pipelines

Para organizar el tratamiento de los datos y su posterior clasificación he decidido utilizar objetos `Pipeline` de `sklearn.pipeline`. 
Estos objetos nos permiten componer una lista de objetos que realizan preprocesado, clasificación o regresión en un único objeto.

En concreto, define un método `fit` que compone los métodos `fit` de sus componentes: llama al método `fit` de un componente, transforma los datos de entrada con `transform` y llama al método `fit` del siguiente componente.

Además, cualquier método del último componente en la lista puede utilizarse, y el objeto `Pipeline` se encargará de transformar los datos usando los componentes de la lista en orden previamente.

El uso de este objeto es común a esta sección y las secciones posteriores.

## Eliminación de datos sin varianza

El primer paso de preprocesado que he decidido hacer es la eliminación de variables que tengan varianza cero, es decir, que sean constantes. 
Estos datos no nos aportan nada en ningún tipo de problema y pueden causar problemas a la hora de comprobar la correlación entre las variables como haremos en las siguientes secciones.

Para este paso utilizamos el objeto `VarianceThreshold` de `sklearn.feature_selection` con un umbral de 0.0 (su valor por defecto).

Aunque podríamos eliminar datos con varianza muy pequeña, en la práctica estos datos pueden ser valiosos y por tanto he decidido dejarlos.

El código puede verse en la sección [Resultado del preprocesamiento].

## Análisis de componentes principales

En el caso del dataset de dígitos tenemos una gran cantidad de dimensiones y por tanto es beneficioso reducir la dimensionalidad para reducir la posibilidad de sobreajuste en los datos de training.

Además, como observamos en la sección [Resultado del preprocesamiento], en ambos datasets hay variables con una correlación alta entre sí. Esto puede dar problemas a la hora de la regresión y otros métodos estadísticos, por lo que es deseable reducir esta correlación [@FarrarMulticollinearityRegressionAnalysis1967].

Debido a estos dos objetivos, he decidido aplicar la técnica de análisis de componentes principales sobre ambos conjuntos de datos.
En concreto, tras hacer la transformación PCA, he conservado variables que expliquen al menos un 95% de la varianza y he descartado el resto.

Para actuar sobre las correlaciones he estandarizado todas las variables (esto es, he hecho que su media sea cero y su varianza 1) usando el objeto `StandardScaler` de `sklearn.preprocessing`.
En el caso de la regresión esto es además necesario porque las unidades de medida utilizada en cada variable son diferentes y no son comparables.

Para realizar esta transformación utilizamos el objeto `PCA` de `sklearn.decomposition`,
con el parámetro `n_components` fijado a `0.95`.

El código de estas dos transformaciones puede verse en la sección [Resultado del preprocesamiento].

## Resultado del preprocesamiento

Para realizar el preprocesado he definido haciendo uso de pipelines un objeto `preprocesador` que realiza el proceso de preprocesado. Para ello defino la lista de pasos en el preprocesado junto con sus nombres con la que inicializo la Pipeline:
```python
preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)
```

Para observar el comportamiento del preprocesado en mi conjunto de variables comparo las matrices de correlación (en valor absoluto) de los datos antes y después del preprocesamiento.
Para ello defino la función `muestra_preprocesado` que muestra ambas matrices (obtenidas con `np.corrcoef`) en un solo plot:

```python
def muestra_preprocesado(datos, procesador, title):
  """Muestra matriz de correlación para datos antes y después del preprocesado."""
  fig, axs = plt.subplots(1, 2, figsize=[12.0, 5.8])

  corr_matrix = np.abs(np.corrcoef(datos.T))
  im = axs[0].matshow(corr_matrix, cmap="plasma")
  axs[0].title.set_text("Sin preprocesado")

  datos_procesados = procesador.fit_transform(datos)
  corr_matrix_post = np.abs(np.corrcoef(datos_procesados.T))
  axs[1].matshow(corr_matrix_post, cmap="plasma")
  axs[1].title.set_text("Con preprocesado")

  fig.suptitle(title)
  fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
  plt.show()
```

El resultado para el dataset de dígitos puede verse en las dos figuras siguientes.

![Matrices de correlación para el conjunto de variables independientes de **dígitos** antes y después del preprocesamiento.](img/preprocesado_optdigits.pdf)
![Matrices de correlación para el conjunto de variables independientes de **airfoil** antes y después del preprocesamiento.](img/preprocesado_airfoil.pdf)

Como vemos la correlación entre las variables se reduce notablemente en ambos casos, siendo casi cero.

Además, en el caso del dataset de dígitos pasamos a tener un total de 40 variables en lugar de las 64 variables, mientras que en el caso de airfoil reducimos en uno el número de variables.

# Selección de clase de hipótesis

## Clasificación

En el problema de clasificación tenemos una gran cantidad de variables.
Esto puede suponer un problema, conocido en la literatura como *curse of dimensionality*, ya que puede dar lugar a sobreajuste o puede hacer que nuestro modelo no converja correctamente. [@TrunkProblemDimensionalitySimple1979]

Para evitar estos problemas he decidido que era más deseable utilizar una clase de hipótesis lineal y no añadir términos de mayor orden, esto es,
$$\mathcal{H}_1 = \left\{ w_0 + \sum_i w_i x_i \;:\; w_i \in \mathbb{R}\right\}.$$

## Regresión

En el caso de la regresión, al contar con una cantidad reducida de variables, optamos por utilizar una clase de hipótesis que incluya términos cuadráticos para así facilitar el ajuste.
En general, es posible que la adición de estos parámetros haga que en este espacio de mayor dimensionalidad la regresión sea más precisa.

La clase de hipótesis queda
$$\mathcal{H}_2 = \left\{ w_0 + \sum_i w_i x_i + \sum_i w'_ix_i^2  \;:\; w_i, w'_i \in \mathbb{R}\right\}.$$

Para llevar a cabo esta transformación utilizamos la clase `FunctionTransformer` del módulo `sklearn.preprocessing` para aplicar una función a los datos. En primer lugar definimos la función a aplicar:
```python
def square(x):
  """Añade variables al cuadrado."""
  return np.hstack((x,x**2))
```

Y procedemos después a hacer un (segundo) escalado para evitar problemas en la regresión.
El conjunto de acciones queda descrito en la siguiente lista:
```python
cuadrado = [("Squaring", FunctionTransformer(func=square)), 
            ("Scaling", StandardScaler())]
```

\newpage

# Conjuntos de training, validación y test

Para separar en training y test 

1. En el primer dataset (`optdigits`), optamos por utilizar la separación con la que nos proveen los autores del paquete. No he encontrado razones para modificarla ya que la distribución de los datos por clases (que viene indicada en la descripción del paquete) parece ser aproximadamente igual en ambos conjuntos de datos.
2. En el segundo dataset (`airfoil`), separamos en training y test usando la función `train_test_split` de `sklearn.model_selection`. Utilizamos un 20% de los datos como test. Para ello llamamos a la función de la siguiente forma
```python
airfoil_tra_x, airfoil_test_x, airfoil_tra_y, airfoil_test_y = train_test_split(
  airfoil_x, airfoil_y, test_size=0.20)
```
donde `airfoil_x` e `airfoil_y` son los datos leídos anteriormente.

Usaremos validación cruzada en el caso de clasificación para estimar los parámetros (ver sección [Selección del modelo lineal]). 

No lo hacemos así en el caso de la regresión ya que en el modelo elegido no tiene sentido hacerlo para la estimación de los parámetros.

# Regularización

He utilizado estrategias de regularización para ambos problemas.
Considero que el uso de la regularización es esencial en el campo del aprendizaje automático en la gran mayoría de problemas, ya que la posibilidad de sobreajuste es muy elevada cuando tratamos con tantas variables de entrada.

El término de regularización controla así la complejidad del modelo resultante, limitando parcialmente este posible sobreajuste en función de cómo de grande sea la constante que acompaña a este término.

Tanto en la regularización del problema de clasificación como en el de regresión he utilizado una regularización de tipo $l_2$, esto es, que añade como término a la función de error un sumando de la forma $\alpha \lVert w\rVert_2^2$, para cierto $\alpha \in \mathbb{R}^+$.

El uso de la regularización de norma 2 penaliza con más fuerza los valores muy grandes en comparación con el uso de la norma 1, lo que considero que es preferible a la hora de limitar la complejidad del modelo.

La elección del parámetro $\alpha$, que en términos prácticos regula el efecto de la regularización difiere en cada caso y se discute en la sección correspondiente de [Selección del modelo lineal].


# Selección del modelo lineal

En esta sección especifico qué modelo lineal se ha escogido para cada problema.

## Clasificación

En el caso del problema de clasificación, me he decantado por un modelo logístico.
El modelo logístico para problemas de clasificación multiclase admite dos posibles estrategias: 

- *One vs. rest*, que resuelve un problema de clasificación binaria para cada clase vs. el resto de clases y
- *Multinomial*, que resuelve el problema de clasificación de todas las clases al mismo tiempo, utilizando una función de pérdida softmax.

Me he decantado por utilizar el segundo método, esto es, la clasificación multinomial que reduce la pérdida de todas las clases al mismo tiempo. 
He añadido, tal y como se menciona en la sección de [regularización], un término de regularización `l2` para evitar el sobreajuste.

Dado que tenemos una cantidad grande de datos, para determinar la constante que acompaña el término regularizador, he utilizado un sistema de validación cruzada con una partición en 5 subconjuntos de datos (de forma que se respete la proporción de clases). El sistema compara el rendimiento para 4 parámetros de regularización entre $10^{-4}$ y $10^4$.

Además, especifico de forma explícita la métrica de error a utilizar y que el modelo lineal incluya una constante adicional. He dejado el proceso de minimización de la función de error con el algoritmo por defecto, ya que el algoritmo recomendado por scikitlearn para problemas con una cantidad de datos que no sea muy grande no está disponible con la función softmax.
El proceso de clasificación queda por tanto indicado en la siguiente lista de tareas:

```python
clasificacion = [("logistic",
                  LogisticRegressionCV(Cs=4,
                                       penalty='l2',
                                       cv=5,
                                       scoring='accuracy',
                                       fit_intercept=True,
                                       multi_class='multinomial'))]
```

El parámetro `scoring='accuracy` se describe en la sección [estimación del error real].

\newpage

## Regresión

Para la regresión, realizamos una regresión lineal (sobre el conjunto de variables que incluyen los términos cuadráticos) utilizando gradiente descendente estocástico con regularización `l2`.

He utilizado como función de error, el error cuadrático (ver sección [Métrica del ajuste] para justificación de esta elección). Además, he fijado el número máximo de iteraciones a 1000 y la tolerancia a $10^{-5}$.

La constante de regularización queda fijada con el valor por defecto (0.0001), ya que he considerado que, puesto que el conjunto de datos no es muy grande, no tenía sentido la utilización de validación cruzada para la obtención del valor óptimo.

Este proceso queda entonces descrito por la siguiente lista de transformaciones:
```python
regresion = [("SGDRegressor",
              SGDRegressor(loss="squared_loss",
                           penalty="l2",
                           max_iter=1000,
                           tol=1e-5))]
```

# Ajuste del modelo final

En esta sección describo el proceso por el cual se entrena el modelo anteriormente descrito con los datos de entrenamiento.

## Clasificación 

En el caso de la clasificación, el proceso que realizamos consiste, en primer lugar, en el preprocesado y a continuación en la clasificación. Utilizando un objeto Pipeline definimos por tanto
```python
clasificador = Pipeline(preprocesado + clasificacion)
```
A continuación entrenamos el modelo. El entrenamiento se realiza utilizando la función `fit` de la Pipeline[^mensaje]:
```python
clasificador.fit(digits_tra_x, digits_tra_y)
```

[^mensaje]: He creado una clase `mensaje` para mostrar un mensaje de espera mientras se realiza el entrenamiento u otros datos. Omito el uso de esta con un bloque `with` en los bloques de código de la memoria ya que no aporta nada nuevo.

Los resultados de la clasificación se discuten en la sección [Discusión].

## Regresión

En el caso de la regresión, el proceso incluye además el añadido de las componentes cuadráticas, por lo que el objeto regresor se define
```python
regresor = Pipeline(preprocesado + cuadrado + regresion)
```
y el ajuste se realiza de forma idéntica con 
```python
regresor.fit(airfoil_tra_x, airfoil_tra_y)
```
Los resultados de la regresión se discuten en la sección de [Discusión].

# Métrica del ajuste

En esta sección describimos qué función de error hemos minimizado durante el proceso de entrenamiento.

## Clasificación

En el caso de la clasificación, como se indicaba en la sección [selección del modelo lineal], he utilizado la función de cross-entropy. 

En concreto, esta función se obtiene a partir de la log-verosimilitud de una distribución multinomial en la que las probabilidades se han modelado con una función softmax[@BishopPatternrecognitionmachine2006a], esto es, si $k \in \{0,\dots,9\}$ es la clase y $\mathbf{x}$ es el vector de características, la probabilidad se modela como 
$$P[y = k | \mathbf{x}] = \frac{\exp(\mathbf{w}_k^T\mathbf{x})}{\sum_{j=0}^9 \exp(\mathbf{w}_j^T\mathbf{x})}.$$

Escogemos entonces vectores $\mathbf{w}_k$ que maximicen por tanto la verosimilitud de estos parámetros en función de la muestra de entrenamiento.

Esta métrica tiene por tanto una justificación teórica robusta, ya que lo que hacemos es aproximar un estimador máximo-verosímil.

A esta log-verosimilitud se le añade también los términos de regularización de la forma $\alpha \lVert \mathbf{w}_k\rVert_2^2$ para el $\alpha$ fijado por el proceso de validación cruzada.

\newpage

## Regresión

En el caso de la regresión he utilizado el MSE, esto es, el error cuadrático medio, dado por (salvo constante)
$$\sum_i (\mathbf{w}^T\mathbf{x}_i - y_i)^2.$$
Otra opción sería el uso del error absoluto medio.
Me he decantado por el error cuadrático dado que este penaliza con mayor severidad los *outliers*: el error crece de forma cuadrática en función de la distancia al valor inferido en lugar de de forma lineal.

Para mostrar el error obtenido sin embargo he mostrado el RMSE (*Root Mean Squared Error*), es decir, la raíz del MSE, para que las unidades de medida coincidan con las de la variable a predecir (en este caso decibelios) y así tengamos una interpretación más adecuada del mismo.

# Estimación del error real

En esta sección describo cómo he medido y obtenido el error en cada problema.

## Clasificación

Para la clasificación, el error es la proporción de ejemplos incorrectamente clasificados.
Esta métrica puede calcularse como uno menos la *accuracy* del modelo.
La asociamos al modelo de regresión logística utilizando el parámetro `scoring='accuracy`, como aparecía en la sección [selección del modelo lineal].

De esta forma el objeto clasificador tendrá una función `score` que, para unos datos `X,y`, nos devolverá la proporción correctamente clasificada de los datos según el modelo (tomando como clasificación correcta la dada por `y`).

Para medir el error de un clasificador por tanto, defino una pequeña función auxiliar que nos permite mostrarlo fácilmente en training y test,
```python
def estima_error_clasif(clasificador, X_tra, y_tra, X_test, y_test, nombre):
  print("Error de {} en training: {:.3f}".format(
    nombre, 1-clasificador.score(X_tra, y_tra)))
  print("Error de {} en test: {:.3f}".format(
    nombre, 1-clasificador.score(X_test, y_test)))
```

Además, para visualizar qué está haciendo el clasificador he definido, a partir de un ejemplo de scikit, una función que muestre la matriz de confusión dado un conjunto de etiquetas reales y predichas.

Esta matriz nos mostrará qué números se confunden más frecuentemente con otros.
Para definirla, nos ayudamos de la función `confusion_matrix` del módulo `sklearn.metrics`.
Normalizamos esta matriz y mostramos en un plot esta matriz, anotando en el centro de cada celda el porcentaje de clasificación redondeado al entero más cercano:

```python
def muestra_confusion(y_real, y_pred, tipo):
  mat = confusion_matrix(y_real, y_pred)
  mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
  fig, ax = plt.subplots()
  ax.matshow(mat, cmap="Purples")
  ax.set(title="Matriz de confusión para predictor {}".format(tipo),
         xticks=np.arange(10),
         yticks=np.arange(10),
         xlabel="Etiqueta real",
         ylabel="Etiqueta predicha")

  for i in range(10):
    for j in range(10):
      ax.text(j,
              i,
              "{:.0f}%".format(mat[i, j]),
              ha="center",
              va="center",
              color="black" if mat[i, j] < 50 else "white")

  plt.show()
```

El resultado del error para este modelo y su matriz de confusión se muestran en la sección de [discusión], junto con la comparación a sus alternativas.

## Regresión

Para la regresión, como mencionamos anteriormente, utilizamos el error cuadrático medio.
Hacemos su raíz cuadrada para facilitar su interpretación.

Además, mostramos también el coeficiente de determinación; un valor en el intervalo $(-\infty,1]$ que vale $1$ cuando el ajuste es perfecto y cuyo valor se reduce en función de cómo de mala sea la clasificación.

Además, tiene la propiedad de que si siempre predecimos la media, este valor valdrá exactamente cero.

Análogamente al apartado anterior, creamos una función que estime estos errores para training y test para un regresor dado,
```python
def estima_error_regresion(regresor, X_tra, y_tra, X_tes, y_tes, nombre):
  print("Errores para regresor {}".format(nombre))
  for datos, X, y in [("training", X_tra, y_tra), ("test", X_tes, y_tes)]:
    y_pred = regresor.predict(X)
    print("  RMSE ({}): {:.3f}".format(
      datos, math.sqrt(mean_squared_error(y, y_pred))))
    print("  R²   ({}): {:.3f}".format(datos, regresor.score(X, y)),
          end="\n\n")
```

Discutimos los resultados y errores obtenidos en la sección [discusión].

# Discusión

Para la discusión he comparado cada modelo con dos modelos alternativos: uno que realiza un ajuste no lineal (Random Forest) y un estimador «dummy» que no realiza realmente ningún ajuste, sino que utiliza alguna estimación como la media o una clasificación estratificada aleatoria.

Creo que estos modelos sirven por un lado como comparación de qué es posible si utilizamos un modelo más complejo y de cuál es el valor de error que obtenemos con los modelos más simples posibles.

## Clasificación

Para la clasificación he comparado con un modelo de random forest y con un clasificador aleatorio estratificado.
El primer clasificador ajusta varios árboles introduciendo aleatoriedad en el conjunto de datos y devuelve la estimación media. Lo definimos con la clase `RandomForestClassifier`, de `sklearn.ensemble`:
```python
randomf_clasif = [("Random Forest", RandomForestClassifier(n_estimators=100))]
clasificador_randomf = Pipeline(preprocesado + randomf_clasif)
```
Fijamos el número de árboles en 100.

El segundo clasificador registra la proporción de clases en training y devuelve aleatoriamente una clase para cada valor con probabilidad proporcional a la presencia de esta clase en training.
Lo definimos con la clase `DummyClassifier`, del módulo `sklearn.dummy`:
```python
dummy_clasif = DummyClassifier(strategy="stratified")
dummy_clasif.fit(digits_tra_x, digits_tra_y)
```

Los resultados son los siguientes, redondeando a 3 cifras decimales

|                         | Logístico | RandomForest | Dummy |
|-------------------------|-----------|--------------|-------|
| $E_{\operatorname{in}}$ | 0.027     | 0.000        | 0.899 |
| $E_{\operatorname{out}}$ | 0.058     | 0.042        | 0.902 |

Como vemos el clasificador logístico consigue clasificar correctamente un ~94% de los datos de test, lo que es un valor muy aceptable para tareas prácticas.

En comparación, el estimador Random Forest consigue un ajuste ligeramente superior, llegando a ajustar perfectamente los datos de training y consiguiendo 0.016 menos de error en test.
No obstante, la complejidad del modelo es muy superior y el paradigma de minimización del riesgo empírico no se sigue en esta clase, ya que la clase de los árboles tiene una dimensión infinita. Por tanto es un modelo más difícil de entrenar y con mayor riesgo de sobreajuste, que creo que no merece la pena utilizar en este caso.

En las siguientes imágenes podemos ver las matrices de confusión para estos dos métodos.

![](img/confusion_logistic.pdf)
![](img/confusion_rf.pdf)

Como vemos, ambos modelos tienen resultado muy parecidos, en los que la clasificación es superior al 85% en todas las clases.
La clase con más clasificaciones incorrectas es la del 8, que es confundida con la clase del 1.
Esto tiene sentido si vemos la representación gráfica obtenida por t-SNE, en la que las clases tienen cierto solapamiento.

El caso base de comparación (*Dummy*), tiene unos resultados muy malos, ya que realmente no está haciendo ningún tipo de clasificación en la práctica: sólo clasifica correctamente en torno a un 10% de los datos, aproximadamente una clasificación aleatoria teniendo en cuenta que los datos se reparten en 10 clases equilibradas.


## Regresión

Para la regresión he comparado con un modelo de random forest y con un regresor que estima la media.
El primer regresor, como en el caso de la clasificación, ajusta varios árboles introduciendo aleatoriedad en el conjunto de datos y devuelve la estimación media. Lo definimos con la clase `RandomForestRegressor`, de `sklearn.ensemble`:
```python
randomf_regr = [("Random Forest", RandomForestRegressor(n_estimators=100))]
regresor_randomf = Pipeline(preprocesado + randomf_regr)
```
Fijamos el número de árboles en 100.

El segundo clasificador registra la proporción de clases en training y devuelve aleatoriamente una clase para cada valor con probabilidad proporcional a la presencia de esta clase en training.
Lo definimos con la clase `DummyRegressor`, del módulo `sklearn.dummy`:
```python
dummy_regression = DummyRegressor(strategy="mean")
dummy_regression.fit(airfoil_tra_x, airfoil_tra_y)
```

Los resultados son los siguientes, redondeando a 3 cifras decimales,

|                  | SGD   | RandomForest | Dummy   |
|------------------|-------|--------------|---------|
| RMSE (training)  | 4.531 | 0.869        | 6.906   |
| $R^2$ (training) | 0.570 | 0.984        | 0.000   |
| RMSE (test)      | 4.302 | 2.178        | 6.859   |
| $R^2$ (test)     | 0.607 | 0.899        | 0.000   |

El clasificador Dummy, que devuelve la media de los datos de training, tiene el valor esperado de coeficiente de determinación tanto en training como en test (lo que nos indica incidentalmente que la partición de training-test elegida es bastante parecida) y tiene un error medio de $6.906$ dB (que se corresponde aproximadamente con la desviación típica de los datos).
Este valor nos sirve como caso de comparación para el resto de clasificadores.

Para el clasificador lineal, la raíz del error obtenida en test es de $4.302 \text{ dB}$, 
un error que podría ser aceptable en aplicaciones prácticas, pero que es aún así bastante alto si comparamos con el error obtenido por el clasificador dummy.
No obstante, podría darse el caso de que los datos tienen una alta variabilidad y por tanto un ajuste muy exacto es imposible (como podría parecer que sugiere la representación gráfica de la visualización).
Su coeficiente de determinación es de 0.570, algo lejano al ajuste perfecto de 1.

El clasificador Random Forest consigue un error mucho menor sin embargo, de aproximadamente la mitad ($2.178$ dB) y un coeficiente de determinación cercano a 1.

Esto parece indicarnos que la clase de funciones que elegimos al comienzo podría no ser la más adecuada: es posible que el ruido producido no sigan una función polinomial de la entrada.
Dependiendo de la aplicación, podría ser preferible entonces el regresor RandomForest u otro no lineal en lugar del elegido, pues, aunque supone un aumento en la complejidad los resultados son notablemente mejores.


\newpage

# Bibliografía {.unnumbered}

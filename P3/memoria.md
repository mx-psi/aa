---
title: Práctica 3
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
lang: es
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
2. el ángulo de ataaque en grados,
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
mkdir -p datos/optdigits
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra \
 -O datos/optdigits/optdigits.tra
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes \
 -O datos/optdigits/optdigits.tes

mkdir -p datos/airfoil
wget \
 archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat \
 -O datos/airfoil/airfoil_self_noise.dat
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
cuadrado = [("Squaring", FunctionTransformer(func=square)), ("Scaling", StandardScaler())]
```

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

Además, especifico de forma explícita la métrica de error a utilizar y que el modelo lineal incluya una constante adicional. El proceso de clasificación queda por tanto indicado en la siguiente lista de tareas:

```python
clasificacion = [("logistic",
                  LogisticRegressionCV(Cs=4,
                                       penalty='l2',
                                       cv=5,
                                       scoring='accuracy',
                                       fit_intercept=True,
                                       multi_class='multinomial'))]
```

## Regresión

# Ajuste del modelo final

# Métrica del ajuste

## Clasificación
## Regresión

En el caso de la regresión he utilizado el MSE.

Para mostrar el error obtenido mostramos el RMSE (*Root Mean Squared Error*), es decir, la raíz del MSE, para que las unidades de medida coincidan con las de la variable a predecir (en este caso decibelios).

# Estimación del error real

# Discusión

Para la discusión he comparado cada modelo con dos modelos alternativos: uno que realiza un ajuste no lineal (Random Forest) y un estimador «dummy» que no realiza realmente ningún ajuste.

## Clasificación

Para la clasificación

## Regresión

\newpage

# Bibliografía {.unnumbered}

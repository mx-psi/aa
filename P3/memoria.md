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
He conservado los nombres de los ficheros originales en todos los casos.

Para crear fácilmente la carpeta de `datos` con la estructura que he utilizado en un sistema Unix pueden ejecutarse los siguientes comandos, también disponibles en el script adjunto `descarga_datos.sh`:

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

Debido a estos dos objetivos, he decidio aplicar la técnica de análisis de componentes principales sobre ambos conjuntos de datos.
En concreto, tras hacer la transformación PCA, he conservado variables que expliquen al menos un 95% de la varianza y he descartado el resto.

Para actuar sobre las correlaciones he estandarizado todas las variables (esto es, he hecho que su media sea cero y su varianza 1) usando el objeto `StandardScaler` de `sklearn.preprocessing`.

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

# Conjuntos de training, validación y test

# Regularización

# Selección del modelo lineal

# Modelo final

# Métrica del ajuste

# Estimación del error real

# Discusión

\newpage

# Bibliografía {.unnumbered}

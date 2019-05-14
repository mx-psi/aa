# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Pablo Baeyens Fernández
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from matplotlib import rcParams

# Fijamos la semilla
np.random.seed(0)
rcParams['axes.titlepad'] = 15

###########################
# CONSTANTES MODIFICABLES #
###########################

DIGITS_TRA = "datos/optdigits/optdigits.tra"
DIGITS_TEST = "datos/optdigits/optdigits.tes"
AIRFOIL = "datos/airfoil/airfoil_self_noise.dat"

##############
# FUNCIONES  #
# AUXILIARES #
##############


def espera():
  input("\n--- Pulsar tecla para continuar ---\n")


def pregunta(pregunta, default="s"):
  """Haz una pregunta de sí o no."""

  if default == "n":
    msg = pregunta + " s/[n]: "
  else:
    msg = pregunta + " [s]/n: "

  while True:
    texto = input(msg).lower()
    texto = default if texto == '' else texto
    if texto.startswith('s') or texto.startswith('y'):
      return True
    elif texto.startswith('n'):
      return False


def imprime_titulo(titulo):
  """Imprime el título de una sección."""
  print("\n" + titulo)
  print("-"*len(titulo), end="\n\n")


def visualiza_clasif(x, y, title=None):
  """Representa conjunto de puntos 2D clasificados.
  Argumentos posicionales:
  - x: Coordenadas 2D de los puntos
  - y: Etiquetas"""

  _, ax = plt.subplots()

  # Establece límites
  xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
  ax.set_xlim(xmin - 1, xmax + 1)
  ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

  # Pinta puntos
  ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

  # Pinta etiquetas
  labels = np.unique(y)
  for label in labels:
    centroid = np.mean(x[y == label], axis=0)
    ax.annotate(int(label),
                centroid,
                size=14,
                weight="bold",
                color="white",
                backgroundcolor="black")

  # Muestra título
  if title is not None:
    plt.title(title)
  plt.show()


def lee_datos(filename, delimiter):
  """Carga datos desde un fichero de texto.
  Argumentos posicionales:
  - filename: Nombre del fichero
  Argumentos opcionales:
  - delimiter: El delimitador que separa los datos
  """
  data = np.loadtxt(filename, delimiter=delimiter)
  return data[:, :-1], data[:, -1]


######################
# OBTENCIÓN DE DATOS #
# (TRAINING Y TEST)  #
######################

imprime_titulo("Obtención de datos")

print("Leyendo datos... ", flush=True, end="")
digitos_tra_x, digitos_tra_y = lee_datos(DIGITS_TRA, delimiter=",")
digitos_test_x, digitos_test_y = lee_datos(DIGITS_TEST, delimiter=",")
airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")
print("Hecho.")

print("Separando training-test... ", flush=True, end="")
airfoil_tra_x, airfoil_test_x, airfoil_tra_y, airfoil_test_y = train_test_split(
  airfoil_x, airfoil_y, test_size=0.25)
print("Hecho.")

##########################
# VISUALIZACIÓN DE DATOS #
##########################

imprime_titulo("Visualización de datos")

print("La visualización de dígitos lleva más de 1 min. (está en la memoria).")

if pregunta("¿Desea generar esta visualización?", default="n"):
  print("Creando visualización...", flush=True, end="")
  X_new = TSNE(n_components=2).fit_transform(digitos_tra_x)
  print("Hecho.")

  visualiza_clasif(X_new,
                   digitos_tra_y,
                   title="Proyección de dígitos en dos dimensiones")

espera()

## FIXME: Pairplots para airfoil

################
# PREPROCESADO #
################

imprime_titulo("Preprocesado")

preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)


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


print("Matriz de correlación pre y post procesado (dígitos)")
muestra_preprocesado(
  VarianceThreshold(threshold=0.0).fit_transform(digitos_tra_x),
  preprocesador,
  title="Problema de clasificación: optdigits")

print("Matriz de correlación pre y post procesado (airfoil)")
muestra_preprocesado(airfoil_tra_x,
                     preprocesador,
                     title="Problema de regresión: airfoil")

#################
# CLASIFICACIÓN #
#################

imprime_titulo("Clasificación")
clasificacion = [("logistic",
                  LogisticRegression(penalty='l2',
                                     solver='sag',
                                     max_iter=400,
                                     multi_class='multinomial'))]

clasificador = Pipeline(preprocesado + clasificacion)

clasificador.fit(digitos_tra_x, digitos_tra_y)
print("Clasificador logístico",
      clasificador.score(digitos_test_x, digitos_test_y))

## REGRESIÓN

###############
# DISCUSIÓN Y #
# ANÁLISIS    #
###############

imprime_titulo("Comparación")

random_forest = [("Random Forest", RandomForestClassifier(n_estimators=100))]

clasificador_random_forest = Pipeline(preprocesado + random_forest)
clasificador_random_forest.fit(digitos_tra_x, digitos_tra_y)
print("Clasificador Random Forest",
      clasificador_random_forest.score(digitos_test_x, digitos_test_y))

# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Pablo Baeyens Fernández
"""
import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegressionCV, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import median_absolute_error
from sklearn.dummy import DummyClassifier, DummyRegressor

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


class mensaje:
  """Clase que gestiona la impresión de mensajes de progreso.
  Se usa con un bloque `with` en el que se introducen las
  órdenes a realizar.
  El bloque NO debe imprimir a pantalla."""

  def __init__(self, mensaje):
    """Indica el mensaje a imprimir."""
    self.mensaje = "> " + mensaje + ": "
    self.en_ejecucion = False
    self.delay = 0.3

  def espera(self):
    i = 0
    wait = ["   ", ".  ", ".. ", "..."]
    while self.en_ejecucion:
      print(self.mensaje + wait[i], end="\r", flush=True)
      time.sleep(self.delay)
      i = (i+1) % 4

  def __enter__(self):
    """Imprime el mensaje de comienzo."""
    print(self.mensaje, end="\r", flush=True)
    self.en_ejecucion = True
    threading.Thread(target=self.espera).start()

  def __exit__(self, tipo, valor, tb):
    """Imprime que ha finalizado la acción."""
    self.en_ejecucion = False
    if tipo is None:
      print(self.mensaje + "hecho.")
    else:
      print("")
      return False


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

with mensaje("Leyendo datos"):
  digits_tra_x, digits_tra_y = lee_datos(DIGITS_TRA, delimiter=",")
  digits_test_x, digits_test_y = lee_datos(DIGITS_TEST, delimiter=",")
  airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")

with mensaje("Separando training-test"):
  airfoil_tra_x, airfoil_test_x, airfoil_tra_y, airfoil_test_y = train_test_split(
    airfoil_x, airfoil_y, test_size=0.25)

##########################
# VISUALIZACIÓN DE DATOS #
##########################

imprime_titulo("Visualización de datos")

print("La visualización de dígitos lleva más de 1 min. (está en la memoria).")

if pregunta("¿Desea generar esta visualización?", default="n"):
  with mensaje("Creando visualización"):
    X_new = TSNE(n_components=2).fit_transform(digits_tra_x)

  visualiza_clasif(X_new,
                   digits_tra_y,
                   title="Proyección de dígitos en dos dimensiones")

espera()

## FIXME: Pairplots para airfoil

################
# PREPROCESADO #
################

imprime_titulo("Preprocesado")


class Duplica:
  """Clase para añadir componentes cuadráticas."""

  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return np.hstack((X, X**2))


preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)


def muestra_preprocesado(datos, procesador, title):
  """Muestra matriz de correlación para datos antes y después del preprocesado."""
  return None  ## FIXME: Borra
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
  VarianceThreshold(threshold=0.0).fit_transform(digits_tra_x),
  preprocesador,
  title="Problema de clasificación: optdigits")

print("Matriz de correlación pre y post procesado (airfoil)")
muestra_preprocesado(airfoil_tra_x,
                     preprocesador,
                     title="Problema de regresión: airfoil")

espera()

#################
# CLASIFICACIÓN #
#################

imprime_titulo("Clasificación")


def estima_error_clasif(clasificador, X, y, nombre):
  print("Accuracy de {}: {:.3f}".format(nombre, clasificador.score(X, y)))


def muestra_confusion(y_real, y_pred, tipo):
  """Muestra matriz de confusión.
  Versión simplificada del ejemplo
  scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
  """
  return None  ## FIXME: Borra
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


clasificacion = [("logistic",
                  LogisticRegressionCV(penalty='l2',
                                       cv=5,
                                       scoring='accuracy',
                                       fit_intercept=True,
                                       multi_class='multinomial'))]

clasificador = Pipeline(preprocesado + clasificacion)

with mensaje("Entrenando modelo de clasificación logístico"):
  clasificador.fit(digits_tra_x, digits_tra_y)

y_pred_logistic = clasificador.predict(digits_test_x)

muestra_confusion(digits_test_y, y_pred_logistic, "Logístico")
estima_error_clasif(clasificador, digits_test_x, digits_test_y, "Logístico")

espera()

#############
# REGRESIÓN #
#############


def estima_error_regresion(regresor, X, y, nombre):
  """Estima diversos errores de un regresor.
  Debe haberse llamado previamente a la función fit del regresor."""
  y_pred = regresor.predict(X)
  print("Errores para regresor {}".format(nombre))
  print("  MedAE: {:.3f}".format(median_absolute_error(y, y_pred)))
  print("  R²: {:.3f}".format(regresor.score(X, y)), end="\n\n")


imprime_titulo("Regresión")

duplicado = [("Duplica", Duplica()), ("scaling", StandardScaler())]
regresion = [("SGDRegressor",
              SGDRegressor(loss="squared_loss",
                           penalty="l2",
                           max_iter=1000,
                           tol=1e-5,
                           shuffle=True))]

regresor = Pipeline(preprocesado + duplicado + regresion)

with mensaje("Ajustando modelo de regresión"):
  regresor.fit(airfoil_tra_x, airfoil_tra_y)

estima_error_regresion(regresor, airfoil_test_x, airfoil_test_y, "SGD")

espera()

###############
# DISCUSIÓN Y #
# ANÁLISIS    #
###############

imprime_titulo("Comparación de clasificación")

randomf_clasif = [("Random Forest", RandomForestClassifier(n_estimators=100))]

clasificador_randomf = Pipeline(preprocesado + randomf_clasif)

with mensaje("Ajustando modelo de clasificación Random Forest"):
  clasificador_randomf.fit(digits_tra_x, digits_tra_y)

y_clasif_randomf = clasificador_randomf.predict(digits_test_x)
muestra_confusion(digits_test_y, y_clasif_randomf, "Random Forest")

estima_error_clasif(clasificador_randomf, digits_test_x, digits_test_y,
                    "RandomForest")

dummy_clasif = DummyClassifier(strategy="stratified")
dummy_clasif.fit(digits_tra_x, digits_tra_y)
estima_error_clasif(dummy_clasif, digits_test_x, digits_test_y,
                    "Estratificado (Dummy)")
espera()

imprime_titulo("Comparación de regresión")

randomf_regr = [("Random Forest", RandomForestRegressor(n_estimators=100))]
regresor_randomf = Pipeline(preprocesado + randomf_regr)

with mensaje("Ajustando modelo de regresión Random Forest"):
  regresor_randomf.fit(airfoil_tra_x, airfoil_tra_y)

estima_error_regresion(regresor_randomf, airfoil_test_x, airfoil_test_y,
                       "RandomForest")

dummy_regression = DummyRegressor(strategy="mean")
dummy_regression.fit(airfoil_tra_x, airfoil_tra_y)
estima_error_regresion(dummy_regression, airfoil_test_x, airfoil_test_y,
                       "Media (dummy)")

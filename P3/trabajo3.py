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

# Fijamos la semilla
np.random.seed(0)

## CONSTANTES MODIFICABLES

DIGITS_TRA = "datos/optdigits/optdigits.tra"
DIGITS_TEST = "datos/optdigits/optdigits.tes"
AIRFOIL = "datos/airfoil/airfoil_self_noise.dat"


def espera():
  input("\n--- Pulsar tecla para continuar ---\n")


def scatter(x,
            y=None,
            style="Classification",
            ws=None,
            labels_ws=None,
            title=None):
  """Representa scatter plot.
    Puede llamarse de 4 formas diferentes

    1. scatter(x)          muestra `x` en un scatter plot
    2. scatter(x,y)        muestra `x` con etiquetas `y` (-1 y 1)
    3. scatter(x,y,ws)     muestra `x` con etiquetas `y` y rectas `ws`
    4. scatter(x,y,ws,lab) muestra `x` con etiquetas `y` y rectas `ws`,
                           etiquetadas por `lab`
    """
  _, ax = plt.subplots()
  xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(np.min(x[:, 1]), np.max(x[:, 1]))

  if y is None:
    ax.scatter(x[:, 0], x[:, 1])
  elif style == "Classification":
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

    labels = np.unique(y)
    for label in labels:
      center = np.mean(x[y == label], axis=0)
      ax.annotate(int(label),
                  center,
                  size=15,
                  weight="bold",
                  color="white",
                  backgroundcolor="black")
  elif style == "Regression":
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", alpha=0.8)

  if ws is not None:
    x = np.array([xmin, xmax])
    if labels_ws is None:
      for w in ws:
        ax.plot(x, (-w[1]*x - w[0])/w[2])
    else:
      for w, name in zip(ws, labels_ws):
        ax.plot(x, (-w[1]*x - w[0])/w[2], label=name)

  if y is not None or ws is not None:
    ax.legend()
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


# ## FIXME: Conveertir en algo usable por el pipeline?
# def eliminador_corr(data, threshold=0.95):
#   """Elimina variables que estén muy correlacionadas entre sí.
#   Argumentos posicionales:
#   - data: Datos de entrada
#   Argumentos opcionales:
#   - threshold: El umbral de correlación a partir del cual eliminar
#   Devuelve:
#   - Los datos con una de las columnas correlacionadas eliminada
#   """

#   corr_matrix = np.corrcoef(data.T)
#   correlated = np.argwhere(corr_matrix > threshold)

#   idxs = list(range(data.shape[1]))
#   for i, j in correlated:
#     if i < j:
#       try:
#         idxs.remove(i)
#       except ValueError:
#         pass

#   def elimina(datos):
#     return datos[:, idxs]

#   return elimina

######################
# OBTENCIÓN DE DATOS #
# (TRAINING Y TEST)  #
######################

digitos_tra_x, digitos_tra_y = lee_datos(DIGITS_TRA, delimiter=",")
digitos_test_x, digitos_test_y = lee_datos(DIGITS_TEST, delimiter=",")

airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")
airfoil_tra_x, airfoil_test_x, airfoil_tra_y, airfoil_test_y = train_test_split(
  airfoil_x, airfoil_y, test_size=0.25)

##########################
# VISUALIZACIÓN DE DATOS #
##########################


def visualiza_datos(X, y, style, title="Visualización"):
  X_new = PCA(n_components=2).fit_transform(X)
  scatter(X_new, y, style, title=title)


visualiza_datos(digitos_tra_x,
                digitos_tra_y,
                style="Classification",
                title="Proyección de dígitos en dos dimensiones")
espera()
visualiza_datos(airfoil_tra_x,
                airfoil_tra_y,
                style="Regression",
                title="Proyección airfoil a dos dimensiones")
espera()

################
# PREPROCESADO #
################

preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

#################
# CLASIFICACIÓN #
#################

clasificacion = [("logistic",
                  LogisticRegression(penalty='l2',
                                     solver='sag',
                                     max_iter=400,
                                     multi_class='multinomial'))]

clasificador = Pipeline(preprocesado + clasificacion)

clasificador.fit(digitos_tra_x, digitos_tra_y)
print(clasificador.score(digitos_test_x, digitos_test_y))

## REGRESIÓN

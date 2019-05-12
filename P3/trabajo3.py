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

## Training y Test

digitos_tra_x, digitos_tra_y = lee_datos(DIGITS_TRA, delimiter=",")
digitos_test_x, digitos_test_y = lee_datos(DIGITS_TEST, delimiter=",")

airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")
airfoil_tra_x, airfoil_test_x, airfoil_tra_y, airfoil_test_y = train_test_split(
  airfoil_x, airfoil_y, test_size=0.25)

## PREPROCESADO

preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

## CLASIFICACIÓN

clasificacion = [("logistic",
                  LogisticRegression(penalty='l2',
                                     solver='sag',
                                     max_iter=400,
                                     multi_class='multinomial'))]

clasificador = Pipeline(preprocesado + clasificacion)

clasificador.fit(digitos_tra_x, digitos_tra_y)
print(clasificador.score(digitos_test_x, digitos_test_y))

## REGRESIÓN

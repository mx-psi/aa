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

# Fijamos la semilla
np.random.seed(0)

## CONSTANTES MODIFICABLES

OPTDIGITS = "datos/optdigits/optdigits.tra"
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


## FIXME: Conveertir en algo de aquí https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
def eliminador_corr(data, threshold=0.95):
  """Elimina variables que estén muy correlacionadas entre sí.
  Argumentos posicionales:
  - data: Datos de entrada
  Argumentos opcionales:
  - threshold: El umbral de correlación a partir del cual eliminar
  Devuelve:
  - Los datos con una de las columnas correlacionadas eliminada
  """

  corr_matrix = np.corrcoef(data.T)
  correlated = np.argwhere(corr_matrix > threshold)

  idxs = list(range(data.shape[1]))
  for i, j in correlated:
    if i < j:
      try:
        idxs.remove(i)
      except ValueError:
        pass

  def elimina(datos):
    return datos[:, idxs]

  return elimina


## PREPROCESADO

digitos_tra_x, digitos_tra_y = lee_datos(OPTDIGITS, delimiter=",")
airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")

## FIXME: ¿Tiene sentido añadir VarianceThreshold ?
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

clasificador = Pipeline(preprocesado)

print(airfoil_x.shape)
clasificador.fit(airfoil_x, airfoil_y)
print(clasificador.transform(airfoil_x).shape)

print(digitos_tra_x.shape)
clasificador.fit(digitos_tra_x, digitos_tra_y)
print(clasificador.transform(digitos_tra_x).shape)

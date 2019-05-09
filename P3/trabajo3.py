# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Pablo Baeyens Fernández
"""
import math
import numpy as np
import matplotlib.pyplot as plt

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


digitos_tra_x, digitos_tra_y = lee_datos(OPTDIGITS, delimiter=",")
airfoil_x, airfoil_y = lee_datos(AIRFOIL, delimiter="\t")


def elimina_corr(data, threshold=0.75):
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
  return data[:, idxs]


print(airfoil_x.shape)
print(elimina_corr(airfoil_x).shape)

print(digitos_tra_x.shape)
print(elimina_corr(digitos_tra_x).shape)

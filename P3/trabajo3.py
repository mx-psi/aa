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

#!/usr/bin/python3
# Autor: Pablo Baeyens Fernández
# Descripción: Implementación del ejercicio 0

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


def espera():
  """Función de espera.
  Evita que la ejecución del programa continue."""
  
  input("Pulse [Enter] para continuar\n")


def parte1(iris):
  """Implementa parte 1 (salvo carga de datos).
  Argumentos posicionales:
  - iris: Dataset iris"""
  
  print("Parte 1")
  print("=======")
  
  # Obtener las características (nombres)
  print("Nombres de las características: ")
  print(iris.feature_names)
  
  # Obtener las características (valores)
  # Imprimo 5 datos como ejemplo
  print("Características (muestra 5 primeros): ")
  print(iris.data[:5])
  
  # Obtener las clases (nombres)
  print("Nombres de las clases: ")
  print(iris.target_names)
  
  # Obtener las clases (valores)
  # Imprimo 5 clases como ejemplo
  print("Clases (muestra 5 primeros):")
  print(iris.target[:5])
  
  # Quedarse con las últimas dos características
  print("Últimas dos características (muestra 5 primeros):")
  plot_data = iris.data[:, -2:]
  print(plot_data[:5])
  
  # Color de cada clase
  class_colors = {0: 'red', 1: 'green', 2: 'blue'}
  
  # Visualizar los datos en un scatter plot
  _, ax = plt.subplots()
  
  # Para cada clase:
  for cls, name in enumerate(iris.target_names):
    # Obten los miembros de la clase
    class_members = plot_data[iris.target == cls]
    
    # Representa en scatter plot
    ax.scatter(
      class_members[:, 0],
      class_members[:, 1],
      c = class_colors[cls],
      label = name)
  
  ax.legend()  # leyenda
  
  print("Muestra en scatter plot (en ventana aparte)")
  espera()  # Espera para ver el resto de datos
  plt.show()


def parte2(iris):
  """Implementa parte 2
  Argumentos posicionales:
  - iris: Dataset iris"""
  
  print("Parte 2")
  print("=======")
  
  # Definimos 4 arrays, inicialmente vacíos
  # `trainX[i]` se corresponde con `trainY[i]`,
  # análogamente para `testX` y `testY`
  
  trainX = np.array([]).reshape((0, 4))
  trainY = np.array([]).reshape((0, 1))
  testX = np.array([]).reshape((0, 4))
  testY = np.array([]).reshape((0, 1))
  
  for cls in range(len(iris.target_names)):
    # Permuta los miembros de una clase
    classX = iris.data[iris.target == cls]
    perm_classX = np.random.permutation(classX)
    
    # Construye el vector de clases
    # Todos los elementos son iguales luego no hace falta permutar
    perm_classY = iris.target[iris.target == cls].reshape((-1, 1))
    
    # Tomamos un 80% de los datos para training
    N = round(len(classX)*0.8)
    
    # Añade 80%-20% a train y test
    trainX = np.vstack((trainX, perm_classX[:N]))
    testX = np.vstack((testX, perm_classX[N:]))
    trainY = np.vstack((trainY, perm_classY[:N]))
    testY = np.vstack((testY, perm_classY[N:]))
  
  print(
    "trainX y trainY (concatenados):\n{}\n".format(
      np.hstack((trainX, trainY))))
  print(
    "testX y testY (concatenados):\n{}\n".format(np.hstack((testX, testY))))
  
  # Cuenta el número en cada clase
  print("Cantidad de cada clase:")
  for cls, name in enumerate(iris.target_names):
    train_number = np.count_nonzero(trainY == cls)
    test_number = np.count_nonzero(testY == cls)
    print(
      " {}: \n  # en training: {}\n  # en test: {}".format(
        name, train_number, test_number))
  espera()


def parte3():
  """Implementa parte 3."""
  
  print("Parte 3")
  print("=======")
  
  # Obtener 100 valores equiespaciados entre 0 y 2π
  x = np.linspace(0, 2*np.pi, num = 100)
  
  # Obtener el valor en x para seno, coseno y su suma
  sin_values = np.sin(x)
  cos_values = np.cos(x)
  sc_values = sin_values + cos_values
  
  # Visualizar las tres curvas simultáneamente en el mismo plot
  plt.plot(
    x,
    sin_values,
    'k--',  # sin
    x,
    cos_values,
    'b--',  # cos
    x,
    sc_values,
    'r--')  # sin + cos
  
  print(
    "Muestra seno (negro), coseno (azul) y suma (rojo) (en ventana aparte)")
  plt.show()


if __name__ == "__main__":
  
  # [Parte 1] Leer la base de datos de iris
  iris = datasets.load_iris()
  
  parte1(iris)
  parte2(iris)
  parte3()

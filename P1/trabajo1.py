# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Pablo Baeyens Fernández
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

# Establece formato unificado
np.set_printoptions(formatter = {'all': lambda x: "{: 1.5f}".format(float(x))})


def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""
  
  def numpy_func(w):
    return func(*w)
  
  return numpy_func


def espera():
  input("\n--- Pulsar Enter para continuar ---\n")


print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')


@to_numpy
def E(u, v):
  """Función E de ejercicio 1.2"""
  return (u**2*np.exp(v) - 2*v**2*np.exp(-u))**2


def dEu(u, v):
  """Derivada parcial de E con respecto a u"""
  return 2 * (u**2 * np.exp(v) - 2 * np.exp(-u) * v**2) * \
      (2 * np.exp(-u) * v**2 + 2 * u * np.exp(v))


def dEv(u, v):
  """Derivada parcial de E con respecto a v"""
  return 2 * (u**2 * np.exp(v) - 4 * np.exp(-u) * v) * \
      (u**2 * np.exp(v) - 2 * np.exp(-u) * v**2)


@to_numpy
def gradE(u, v):
  """Gradiente de E"""
  return np.array([dEu(u, v), dEv(u, v)])


###########################
# Ejercicio 1. Apartado 1 #
###########################


def gradient_descent(
    initial_point, fun, grad_fun, eta, max_iter, error2get = -math.inf):
  """ Aproxima el mínimo de una función mediante
    el método de gradiente descendiente.
    Argumentos posicionales:
    - initial_point: Punto inicial
    - fun: Función a minimizar. Debe ser diferenciable
    - grad_fun: Gradiente de `fun`
    - eta: Tasa de aprendizaje
    - max_iter: Máximo número de iteraciones

    Argumentos opcionales:
    - error2Get: error en caso de criterio de tolerancia.
    Si no se especifica se ignora este criterio de parada.

    Devuelve:
    - Mínimo hallado
    - Número de iteraciones que se han necesitado
    (`max_iter` si no se pasa `error2Get`)
    """
  
  w = initial_point
  iterations = 0
  
  while fun(w) > error2get and iterations < max_iter:
    w = w - eta*grad_fun(w)
    iterations += 1
  
  return w, iterations


############################
# Ejercicio 1. Apartado 2. #
############################

eta = 0.01
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0, 1.0])
E_minima, it = gradient_descent(
  initial_point, E, gradE, eta, maxIter, error2get)

print('Numero de iteraciones: {}'.format(it))
print('Coordenadas obtenidas: {}'.format(E_minima))
print('Valor de la función en el punto: {}'.format(E(E_minima)))


# DISPLAY FIGURE
def display_figure():
  """Muestra figura 3D del punto mínimo hallado para el ejercicio 1"""
  x = np.linspace(-50, 50, 50)
  y = np.linspace(-50, 50, 50)
  X, Y = np.meshgrid(x, y)
  Z = E([X, Y])  # E_w([X, Y])
  fig = plt.figure()
  ax = Axes3D(fig)
  surf = ax.plot_surface(
    X,
    Y,
    Z,
    edgecolor = 'none',
    rstride = 1,
    cstride = 1,
    cmap = 'jet',
    alpha = 0.8)
  min_point = np.array([E_minima[0], E_minima[1]])
  min_point_ = min_point[:, np.newaxis]
  ax.plot(
    min_point_[0],
    min_point_[1],
    E([min_point_[0], min_point_[1]]),
    'r*',
    markersize = 5)
  ax.set(
    title =
    'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
  ax.set_xlabel('u')
  ax.set_ylabel('v')
  ax.set_zlabel('E(u,v)')
  plt.show()


display_figure()
espera()

############################
# Ejercicio 1. Apartado 3. #
############################


@to_numpy
def f(x, y):
  """Función f de ejercicio 1.3"""
  return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


def dfx(x, y):
  """Derivada parcial de f respecto de x."""
  return 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)


def dfy(x, y):
  """Derivada parcial de f respecto de y."""
  return 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)


@to_numpy
def gradf(x, y):
  """Gradiente de f"""
  return np.array([dfx(x, y), dfy(x, y)])


maxIter = 50
initial_point = np.array([0.1, 0.1])

# 1.3 a)

# Obten resultados en cada iteración
resEtaPeq = np.empty((maxIter, ))
resEtaGrande = np.empty((maxIter, ))

for eta, resultados in [(0.01, resEtaPeq), (0.1, resEtaGrande)]:
  w = initial_point
  iterations = 0
  
  while iterations < maxIter:
    # Guarda el resultado en la iteración actual
    resultados[iterations] = f(w)
    w = w - eta*gradf(w)
    iterations += 1


def compara_resultados():
  """Muestra curvas de decrecimiento para GD con diferentes tasas de aprendizaje."""
  print("Curvas de decrecimiento para el gradiente descendente")
  plt.plot(resEtaPeq, 'b-o', label = r"$\eta$ = 0.01")
  plt.plot(resEtaGrande, 'k-o', label = r"$\eta$ = 0.1")
  plt.legend()
  plt.show()


compara_resultados()
espera()

# 1.3 b)

# Puntos iniciales
initial_points = map(np.array, [[0.1, 0.1], [1, 1], [-.5, -.5], [-1, -1]])

# Imprime punto inicial, punto final y valor en un punto
print("{:^17}  {:^17}  {:^9}".format("Inicial", "Final", "Valor"))
for initial in initial_points:
  w, _ = gradient_descent(initial, f, gradf, 0.01, 50)
  print("{}  {}  {: 1.5f}".format(initial, w, f(w)))

# Halla mejor mínimo
eta = 0.01
f_minima, _ = gradient_descent(initial_point, f, gradf, eta, maxIter)


def contour_plot(min_point):
  """Gráfica de contorno del punto mínimo hallado para f y la curva que define
    la región en la que están los mínimos globales."""
  x = np.arange(-2, 2, 0.01)
  y = np.arange(-2, 2, 0.01)
  xx, yy = np.meshgrid(x, y, sparse = True)
  z = f([xx, yy])
  h = plt.contourf(x, y, z, cmap = "plasma")
  plt.contour(x, y, xx**2 + 2*yy**2 - 2, [0])
  plt.plot(min_point[0], min_point[1], 'r*', markersize = 5)
  plt.show()


contour_plot(f_minima)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
espera()
print('\nEJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos


def readData(file_x, file_y):
  # Leemos los ficheros
  datax = np.load(file_x)
  datay = np.load(file_y)
  y = []
  x = []
  # Solo guardamos los datos cuya clase sea la 1 o la 5
  for i in range(0, datay.size):
    if datay[i] == 5 or datay[i] == 1:
      if datay[i] == 5:
        y.append(label5)
      else:
        y.append(label1)
      x.append(np.array([1, datax[i][0], datax[i][1]]))
  
  x = np.array(x, np.float64)
  y = np.array(y, np.float64)
  
  return x, y


# Funcion para calcular el error


def Err(x, y, w):
  """Calcula el error para un modelo de regresión lineal"""
  wN = np.linalg.norm(x.dot(w) - y)**2
  return wN/len(x)


def dErr(x, y, w):
  """Calcula derivada de error para un modelo de regresión lineal."""
  return 2/len(x)*(x.T.dot(x.dot(w) - y))


def scatter(x, y = None, ws = None, labels_ws = None):
  """Representa scatter plot.
    Puede llamarse de 4 formas diferentes

    1. scatter(x)          muestra `x` en un scatter plot
    2. scatter(x,y)        muestra `x` con etiquetas `y` (-1 y 1)
    3. scatter(x,y,ws)     muestra `x` con etiquetas `y` y rectas `ws`
    4. scatter(x,y,ws,lab) muestra `x` con etiquetas `y` y rectas `ws`,
                           etiquetadas por `lab`
    """
  
  _, ax = plt.subplots()
  xmin, xmax = np.min(x[:, 1]), np.max(x[:, 1])
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(np.min(x[:, 2]), np.max(x[:, 2]))
  
  if y is None:
    ax.scatter(x[:, 1], x[:, 2])
  else:
    class_colors = {-1: 'green', 1: 'blue'}
    # Para cada clase:
    for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
      # Obten los miembros de la clase
      class_members = x[y == cls]
      
      # Representa en scatter plot
      ax.scatter(
        class_members[:, 1],
        class_members[:, 2],
        c = class_colors[cls],
        label = name)
  
  if ws is not None:
    x = np.array([xmin, xmax])
    if labels_ws is None:
      for w in ws:
        ax.plot(x, (-w[1]*x - w[0])/w[2])
    else:
      for w, name in zip(ws, labels_ws):
        ax.plot(x, (-w[1]*x - w[0])/w[2], label = name)
  
  if y is not None or ws is not None:
    ax.legend()
  plt.show()


# # Gradiente Descendente Estocástico
def sgd(x, y, eta = 0.01, max_iter = 1000, batch_size = 32):
  """Implementa la función de gradiente descendiente estocástico
    para problemas de regresión lineal.
    Argumentos posicionales:
    - x: Datos en coordenadas homogéneas
    - y: Etiquetas asociadas (-1 o 1)
    Argumentos opcionales:
    - eta: Tasa de aprendizaje
    - max_iter: máximo número de iteraciones
    - batch_size: tamaño del batch"""
  
  w = np.zeros((3, ))
  iterations = 0
  
  idxs = np.arange(len(x))  # vector de índices
  batch_start = 0  # Comienzo de la muestra
  
  while iterations < max_iter:
    if batch_start == 0:  # Si empezamos una época, shuffle
      idxs = np.random.permutation(idxs)
    # Toma índices
    idx = idxs[batch_start:batch_start + batch_size]
    
    w = w - eta*dErr(x[idx, :], y[idx], w)
    iterations += 1
    
    # Actualiza el comienzo del batch
    batch_start += batch_size
    if batch_start > len(x):  # Si hemos llegado al final reinicia
      batch_start = 0
  return w


# Pseudo-inversa
def pseudoinverse(x, y):
  """Calcula el vector w a partir del método de la pseudo-inversa."""
  u, s, v = np.linalg.svd(x)
  d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

w_sgd = sgd(x, y, eta = 0.01, max_iter = 20000)
print('Bondad del resultado para grad. descendente estocastico:')
print("  Ein:  ", Err(x, y, w_sgd))
print("  Eout: ", Err(x_test, y_test, w_sgd))

w_pinv = pseudoinverse(x, y)
print('\nBondad del resultado para pseudo-inversa:')
print("  Ein:  ", Err(x, y, w_pinv))
print("  Eout: ", Err(x_test, y_test, w_pinv))

espera()
print("\nGráfica de resultados de SGD y pseudoinversa")

scatter(x, y, [w_sgd, w_pinv], ["SGD", "Pinv"])
espera()

# Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]


def simula_unif(N, d, size):
  return np.random.uniform(-size, size, (N, d))


@to_numpy
def f_label(x1, x2):
  """Función de apartado 2.2.b"""
  return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)


def genera_datos():
  """Genera datos aleatorios para el experimento."""
  # Genera datos aleatorios
  x = simula_unif(1000, 2, 1)
  
  # Genera etiquetas (por f y por ruido)
  y_f = f_label(x[:900, :].T)
  y_rand = np.random.choice([-1, 1], 100)
  y = np.hstack((y_f, y_rand))
  
  return x, y


def genera_hom():
  """Genera los datos en coordenadas homogéneas"""
  x, y = genera_datos()
  x_hom = np.hstack((np.ones((1000, 1)), x))
  return x_hom, y


# EXPERIMENTO. Apartado a (y etiquetas de b)
x, y = genera_hom()
print("Puntos aleatorios generados")
scatter(x)
espera()

# EXPERIMENTO. Apartado b representación
print("Puntos etiquetados")
scatter(x, y)
espera()

w = sgd(x, y)
print("Bondad del resultado del experimento (1 ejecución)")
print("  Ein:  {}".format(Err(x, y, w)))


# EXPERIMENTO. Apartado d) Experimento
def experimento():
  """Experimento del apartado d)"""
  x, y = genera_hom()
  x_test, y_test = genera_hom()
  w = sgd(x, y)
  Ein = Err(x, y, w)
  Eout = Err(x_test, y_test, w)
  return np.array([Ein, Eout])


# Haz 1000 ejecuciones del experimento
Nexp = 1000
errs = 0
for _ in range(Nexp):
  errs = errs + experimento()
Ein_medio, Eout_medio = errs/Nexp

print("Bondad del resultado del experimento ({} ejecuciones)".format(Nexp))
print("  Ein:  {}".format(Ein_medio))
print("  Eout: {}".format(Eout_medio))
espera()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print("BONUS")


@to_numpy
def hessianf(x, y):
  return np.array(
    [
      2 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
      8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
      8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
      4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    ]).reshape((2, 2))


def newton(initial_point, fun, grad_fun, hessian, eta, max_iter):
  """ Aproxima mínimos con el método de Newton.
    Argumentos posicionales:
    - initial_point: Punto inicial
    - fun: Función a minimizar. Debe ser diferenciable
    - grad_fun: Gradiente de `fun`
    - hessian: hessiana de la función
    - eta: Tasa de aprendizaje
    - max_iter: Máximo número de iteraciones

    Devuelve:
    - Mínimo hallado
    """
  
  w = initial_point
  w_list = [initial_point]
  iterations = 0
  
  while iterations < max_iter:
    w = w - eta*np.linalg.inv(hessian(w)).dot(grad_fun(w))
    w_list.append(w)
    iterations += 1
  
  return np.array(w_list)


# Representación de curva de decrecimiento

print("Representación de curva de decrecimiento del método de Newton")
resEtaPeqNewton = np.apply_along_axis(
  f, 1, newton(np.array([0.1, 0.1]), f, gradf, hessianf, 0.01, 50))
resEtaGrandeNewton = np.apply_along_axis(
  f, 1, newton(np.array([0.1, 0.1]), f, gradf, hessianf, 0.1, 50))

plt.plot(resEtaPeq, 'b-o', label = r"GD, $\eta$ = 0.01")
plt.plot(resEtaGrande, 'k-o', label = r"GD, $\eta$ = 0.1")
plt.plot(resEtaPeqNewton, 'g-o', label = r"Newton, $\eta$ = 0.01")
plt.plot(resEtaGrandeNewton, 'c-o', label = r"Newton, $\eta$ = 0.1")
plt.legend()
plt.show()

espera()

print("Ejecución del método de Newton para diversos puntos iniciales:\n")
# Cálculo de puntos del método de Newton
initial_points = zip(
  ["0.1, 0.1", "1,1", "-0.5, -0.5", "-1, -1"],
  map(np.array, [[0.1, 0.1], [1, 1], [-.5, -.5], [-1, -1]]))
print("Punto inicial, punto final, valor final, gradiente final:")
steps_for = {}
for label, initial in initial_points:
  points = newton(initial, f, gradf, hessianf, 0.01, 800)
  steps_for[label] = points
  print(
    "{:>10}".format(label), points[-1, :], "{: 1.5f}".format(f(points[-1, :])),
    gradf(points[-1, :]))

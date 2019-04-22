# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Pablo Baeyens Fernández
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(274)


def espera():
  input("\n--- Pulsar tecla para continuar ---\n")


def simula_unif(N, dim, rango):
  return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
  media = 0
  out = np.zeros((N, dim), np.float64)
  for i in range(N):
    # Para cada columna dim se emplea un sigma determinado. Es decir, para
    # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
    # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
    out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

  return out


def simula_recta(intervalo):
  points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
  x1 = points[0, 0]
  x2 = points[1, 0]
  y1 = points[0, 1]
  y2 = points[1, 1]
  # y = a*x + b
  a = (y2-y1)/(x2-x1)  # Calculo de la pendiente.
  b = y1 - a*x1  # Calculo del termino independiente.

  return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida
# correspondiente


def scatter(x, y=None, ws=None, labels_ws=None, title=None):
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
  else:
    class_colors = {-1: 'green', 1: 'blue'}
    # Para cada clase:
    for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
      # Obten los miembros de la clase
      class_members = x[y == cls]

      # Representa en scatter plot
      ax.scatter(class_members[:, 0],
                 class_members[:, 1],
                 c=class_colors[cls],
                 label=name)

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


# Apartado a)
print("Apartado 1.1.a (en ventana aparte)")
x = simula_unif(50, 2, [-50, 50])
scatter(x, title="Nube de puntos uniforme")

# Apartado b)
print("Apartado 1.1.b (en ventana aparte)")
x = simula_gaus(50, 2, np.array([5, 7]))
scatter(x, title="Nube de puntos gaussiana")

espera()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida
# correspondiente


# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
  if x >= 0:
    return 1
  return -1


def f(x, y, a, b):
  return signo(y - a*x - b)


# CODIGO DEL ESTUDIANTE
intervalo = [-50, 50]
a, b = simula_recta(intervalo)
vector_recta = np.array([b, a, -1])

N = 50
x = simula_unif(N, 2, intervalo)
x_hom = np.hstack((np.ones((N, 1)), x))

y = np.empty((N, ))
for i in range(N):
  y[i] = f(x[i, 0], x[i, 1], a, b)

print("Apartado 1.2.a (ventana aparte)")
scatter(x,
        y,
        ws=[vector_recta],
        labels_ws=["Frontera"],
        title="Puntos etiquetados en función de recta aleatoria")

espera()

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello

y_noise = y.copy()

# Modifica un 10% aleatorio de cada etiqueta
for label in {-1, 1}:
  y_lab = np.nonzero(y == label)[0]
  y_rand = np.random.choice(y_lab, math.ceil(0.1*len(y_lab)), replace=False)
  y_noise[y_rand] = -y_noise[y_rand]

print("Apartado 1.2.b (ventana aparte)")
scatter(x,
        y_noise,
        ws=[vector_recta],
        labels_ws=["Frontera"],
        title="Puntos etiquetados con recta aleatoria (con ruido)")

espera()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la
# frontera de clasificación de los puntos de la muestra en lugar de una
# recta


def plot_datos_cuad(X,
                    y,
                    fz,
                    title='Point cloud plot',
                    xaxis='x axis',
                    yaxis='y axis'):
  # Preparar datos
  min_xy = X.min(axis=0)
  max_xy = X.max(axis=0)
  border_xy = (max_xy-min_xy)*0.01

  # Generar grid de predicciones
  xx, yy = np.mgrid[min_xy[0] - border_xy[0]:max_xy[0] + border_xy[0] +
                    0.001:border_xy[0], min_xy[1] - border_xy[1]:max_xy[1] +
                    border_xy[1] + 0.001:border_xy[1]]
  grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
  pred_y = fz(grid)
  # pred_y[(pred_y>-1) & (pred_y<1)]
  pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

  # Plot
  f, ax = plt.subplots(figsize=(8, 6))
  contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
  ax_c = f.colorbar(contour)
  ax_c.set_label('$f(x, y)$')
  ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
  ax.scatter(X[:, 0],
             X[:, 1],
             c=y,
             s=50,
             linewidth=2,
             cmap="RdYlBu",
             edgecolor='white')

  XX, YY = np.meshgrid(
    np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]),
    np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]))
  positions = np.vstack([XX.ravel(), YY.ravel()])
  ax.contour(XX,
             YY,
             fz(positions.T).reshape(X.shape[0], X.shape[0]), [0],
             colors='black')

  ax.set(xlim=(min_xy[0] - border_xy[0], max_xy[0] + border_xy[0]),
         ylim=(min_xy[1] - border_xy[1], max_xy[1] + border_xy[1]),
         xlabel=xaxis,
         ylabel=yaxis)
  plt.title(title)
  plt.show()


# CODIGO DEL ESTUDIANTE


def getPorc(datos, labels, clasificador):
  """Obtiene el porcentaje de puntos correctamente clasificados
    por un clasificador dado.
    Argumentos posicionales:
    - datos: datos,
    - labels: etiquetas,
    - clasificador: Clasificador"""

  # Los campos no negativos indican clasificación correcta
  signos = labels*clasificador(datos)
  return 100*len(signos[signos >= 0])/len(labels)


# Lista de clasificadores con su título
clasificadores = [
  (lambda x: x[:, 1] - a*x[:, 0] - b, "Recta"),
  (lambda x: (x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 1"),
  (lambda x: 0.5*(x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 2"),
  (lambda x: 0.5*(x[:, 0] - 10)**2 + (x[:, 1] + 20)**2 - 400, "Elipse 3"),
  (lambda x: x[:, 1] - 20*x[:, 0]**2 - 5*x[:, 0] + 3, "Parábola")
]

print("Apartado 1.3 (ventana aparte y terminal)")

# Representa y calcula el número bien clasificado para cada tipo
for clasif, title in clasificadores:
  plot_datos_cuad(x, y_noise, clasif, title=title)
  print("Correctos para '{}': {}%".format(title, getPorc(x, y_noise, clasif)))

espera()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON


def ajusta_PLA(datos, labels, max_iter, vini):
  """Calcula el hiperplano solución al problema de clasificación binaria.
    Argumentos posicionales:
    - datos: matriz de datos,
    - labels: Etiquetas,
    - max_iter: Número máximo de iteraciones
    - vini: Valor inicial
  Devuelve:
  - w, El vector de pesos y
  - iterations el número de iteraciones."""

  w = vini.copy()

  for it in range(1, max_iter + 1):
    w_old = w.copy()

    for dato, label in zip(datos, labels):
      if signo(w.dot(dato)) != label:
        w += label*dato

    if np.all(w == w_old):  # No hay cambios
      return w, it

  return w, it


def clasifHiperplano(w):
  """Devuelve un clasificador
  respecto del vector normal de un hiperplano"""
  return lambda x: x.dot(w)


def testPLA(x, y, max_iters=1000):
  """Prueba el algoritmo de Perceptron para un conjunto de x dado."""

  print("Vector inicial cero")
  w, its = ajusta_PLA(x, y, max_iters, np.zeros(3))

  print("Iteraciones: {} épocas".format(its))
  print("% correctos: {}%".format(getPorc(x, y, clasifHiperplano(w))))

  print("Vector inicial aleatorio (media de 10 ejecuciones)")

  iterations = np.empty((10, ))
  percentages = np.empty((10, ))

  for i in range(10):
    w, its = ajusta_PLA(x, y, max_iters, np.random.rand(3))
    iterations[i] = its
    percentages[i] = getPorc(x, y, clasifHiperplano(w))

  print('Iteraciones: {} épocas (± {:.02f})'.format(np.mean(iterations),
                                                    np.std(iterations)))
  print('% correctos: {}% (± {:.02f})'.format(np.mean(percentages),
                                              np.std(percentages)))


print("Apartado 2.1.a (sin ruido)")
testPLA(x_hom, y)
espera()

# Ahora con los datos del ejercicio 1.2.b

print("Apartado 2.1.b (con ruido)")
testPLA(x_hom, y_noise)
espera()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT


def gradRL(dato, label, w):
  return -label*dato/(1 + np.exp(label*w.dot(dato)))


def sgdRL(datos, labels, eta=0.01):
  """Implementa el algoritmo de regresión logística
  mediante SGD con tamaño de batch 1.
  Argumentos posicionales:
  - datos: datos y
  - labels: etiquetas.
  Devuelve: Vector w que define el clasificador.
  """

  N, dim = datos.shape
  w = np.zeros(dim)
  ha_cambiado = True  # Si ha variado en la época actual
  idxs = np.arange(N)  # vector de índices

  while ha_cambiado:
    w_old = w.copy()
    idxs = np.random.permutation(idxs)
    for idx in idxs:
      w += -eta*gradRL(datos[idx], labels[idx], w)
    ha_cambiado = np.linalg.norm(w - w_old) > 0.01

  return w


intervalo = [-2, 2]
a, b = simula_recta(intervalo)

N = 100
datos = simula_unif(N, 2, intervalo)
datos_hom = np.hstack((np.ones((N, 1)), datos))

labels = np.empty((N, ))
for i in range(N):
  labels[i] = f(datos[i, 0], datos[i, 1], a, b)

w = sgdRL(datos_hom, labels)

print("Apartado 2.b (en ventana aparte)")
scatter(datos, labels, ws=[w], labels_ws=["RL"])
espera()

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


def ErrRL(w, x, y):
  """Calcula el error de un clasificador logístico para una muestra de datos.
  Argumentos opcionales:
  - w: Vector de pesos del clasificador logístico,
  - x: datos en coordenadas homogéneas,
  - y: etiquetas
  Devuelve:
  - El error logístico"""
  return np.mean(np.log(1 + np.exp(-y*x.dot(w))))


N_test = 1000
test = simula_unif(N_test, 2, intervalo)
test_hom = np.hstack((np.ones((N_test, 1)), test))

labels_test = np.empty((N_test, ))
for i in range(N_test):
  labels_test[i] = f(test[i, 0], test[i, 1], a, b)

print("% correctos en test RL: {}%".format(
  getPorc(test_hom, labels_test, clasifHiperplano(w))))
print("Error: {}".format(ErrRL(w, test_hom, labels_test)))

espera()

###############################################################################
###############################################################################
###############################################################################
# BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
  # Leemos los ficheros
  datax = np.load(file_x)
  datay = np.load(file_y)
  y = []
  x = []
  # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
  for i in range(0, datay.size):
    if datay[i] == digits[0] or datay[i] == digits[1]:
      if datay[i] == digits[0]:
        y.append(labels[0])
      else:
        y.append(labels[1])
      x.append(np.array([1, datax[i][0], datax[i][1]]))

  x = np.array(x, np.float64)
  y = np.array(y, np.float64)

  return x, y


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4, 8], [-1, 1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4, 8],
                          [-1, 1])

# LINEAR REGRESSION FOR CLASSIFICATION


def Err(x, y, w):
  """Calcula el error para un modelo de regresión lineal"""
  wN = np.linalg.norm(x.dot(w) - y)**2
  return wN/len(x)


# Pseudo-inversa
def pseudoinverse(x, y):
  """Calcula el vector w a partir del método de la pseudo-inversa."""
  u, s, v = np.linalg.svd(x)
  d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)


print("Apartado 3.2 Cálculo de regresión lineal...", end="", flush=True)
w_lin = pseudoinverse(x, y)
print("Hecho.")

# POCKET ALGORITHM


def PLAPocket(datos, labels, max_iter, vini):
  """Calcula el hiperplano solución al problema de clasificación binaria.
    Argumentos posicionales:
    - datos: matriz de datos,
    - labels: Etiquetas,
    - max_iter: Número máximo de iteraciones
    - vini: Valor inicial
  Devuelve:
  - w, El vector de pesos y
  - iterations el número de iteraciones."""
  w = vini.copy()
  w_best = w.copy()
  err_best = Err(datos, labels, w_best)

  for it in range(1, max_iter + 1):
    w_old = w.copy()

    for dato, label in zip(datos, labels):
      if signo(w.dot(dato)) != label:
        w += label*dato

    err = Err(datos, labels, w)
    if err < err_best:
      w_best = w.copy()
      err_best = err

    if np.all(w == w_old):  # No hay cambios
      return w_best, it

  return w_best, it


print("Apartado 3.2. PLAPocket a partir de RL...", end="", flush=True)
w_pla, _ = PLAPocket(x, y, 1000, w_lin)
print("Hecho")

print("Apartado 3.2. PLAPocket a partir de vector de ceros...",
      end="",
      flush=True)
w_pla_cero, _ = PLAPocket(x, y, 1000, np.random.rand(3))
print("Hecho")

rectas = [w_lin, w_pla, w_pla_cero]
nombres = ["Regresión lineal", "PLA-Pocket (RL)", "PLA-Pocket (random)"]

print("Apartado 3.2.a")

scatter(x[:, 1:],
        y,
        ws=rectas,
        labels_ws=nombres,
        title='Digitos Manuscritos (TRAINING) con rectas estimadas')

scatter(x_test[:, 1:],
        y_test,
        ws=rectas,
        labels_ws=nombres,
        title='Digitos Manuscritos (TEST) con rectas estimadas')

espera()

# ERRORES obtenidos

for w_recta, nombre in zip(rectas, nombres):
  print("Ein   para {:^18}: {}".format(nombre, Err(x, y, w_recta)))
  print("Etest para {:^18}: {}".format(nombre, Err(x_test, y_test, w_recta)))

espera()

# COTA SOBRE EL ERROR


def cota(err, N, delta):
  """Calcula cota superior de Eout.
  Argumentos posicionales:
  - err: El error estimado,
  - N: El tamaño de la muestra y
  - delta: La tolerancia a error.
  Devuelve:
  - Cota superior de Eout"""
  return err + np.sqrt(1/(2*N)*(np.log(2/delta) + 3*64*np.log(2)))


Ein = Err(x, y, w_pla)
Etest = Err(x_test, y_test, w_pla)
Nin = len(x)
Ntest = len(x_test)
delta = 0.05

print("Apartado 3.2.c (en terminal)")
print("Cota superior de Eout   (con Ein): {}".format(cota(Ein, Nin, delta)))
print("Cota superior de Eout (con Etest): {}".format(cota(Etest, Ntest,
                                                          delta)))

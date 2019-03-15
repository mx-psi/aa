# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Pablo Baeyens Fernández
"""

import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(1)

# Establece formato unificado
np.set_printoptions(formatter={'all':lambda x: "{: 1.5f}".format(float(x))})


def espera():
  input("\n--- Pulsar Enter para continuar ---\n")

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

def E(u,v):
  """Función E de ejercicio 1.2"""
  return (u**2*np.exp(v) - 2*v**2*np.exp(-u))**2

def dEu(u,v):
  """Derivada parcial de E con respecto a u"""
  return 2*(u**2*np.exp(v) - 2*np.exp(-u)*v**2)*(2*np.exp(-u)*v**2 + 2*u*np.exp(v))

def dEv(u,v):
  """Derivada parcial de E con respecto a v"""
  return 2*(u**2*np.exp(v) - 4*np.exp(-u)*v)*(u**2*np.exp(v) - 2*np.exp(-u)*v**2)

def gradE(u,v):
  """Gradiente de E"""
  return np.array([dEu(u,v), dEv(u,v)])


###########################
# Ejercicio 1. Apartado 1 #
###########################

def gradient_descent(initial_point, fun, grad_fun, eta, max_iter, error2get = -math.inf):
  """ Calcula la función de gradiente descendiente.
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

  while fun(*w) > error2get and iterations < max_iter:
    w = w - eta*grad_fun(*w)
    iterations += 1

  return w, iterations


############################
# Ejercicio 1. Apartado 2. #
############################

eta = 0.01
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point, E, gradE, eta, maxIter, error2get)


print('Numero de iteraciones: {}'.format(it))
print('Coordenadas obtenidas: {}'.format(w))
print('Valor de la función en el punto: {}'.format(E(*w)))


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-50, 50, 50)
y = np.linspace(-50, 50, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet',alpha=0.8)
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=5)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

plt.show()

espera()



############################
# Ejercicio 1. Apartado 3. #
############################


def f(x,y):
  """Función f de ejercicio 1.3"""
  return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def dfx(x,y):
  """Derivada parcial de f respecto de x."""
  return 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)

def dfy(x,y):
  """Derivada parcial de f respecto de y."""
  return 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

def gradf(x,y):
  """Gradiente de f"""
  return np.array([dfx(x,y), dfy(x,y)])


eta = 0.01
maxIter = 50
initial_point = np.array([0.1,0.1])

## 1.3 a)
min_point, it = gradient_descent(initial_point, f, gradf, eta, maxIter)
print(min_point)

## 1.3 b)

# Puntos iniciales
initial_points = map(np.array, [[0.1,0.1], [1,1], [-.5, -.5], [-1, -1]])

# Imprime punto inicial, punto final y valor en un punto
print("{:^17}  {:^17}  {:^9}".format("Inicial","Final","Valor"))
for initial in initial_points:
  w, _ = gradient_descent(initial, f, gradf, eta, maxIter)
  print("{}  {}  {: 1.5f}".format(initial, w, f(*w)))


x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)
xx, yy = np.meshgrid(x, y, sparse=True)
z = f(xx,yy)
h = plt.contourf(x,y,z, cmap="plasma")
plt.contour(x,y, xx**2 + 2*yy**2 - 2, [0])
plt.plot(min_point[0], min_point[1], 'r*', markersize=5)
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
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
	for i in range(0,datay.size):
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
def Err(x,y,w):
  return

# # Gradiente Descendente Estocastico
# def sgd(?):
#   #
#   return w

# # Pseudoinversa
# def pseudoinverse(?):
#   #
#   return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


#w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...

# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Pablo Baeyens Fernández
"""
import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(2)


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
    a = (y2 - y1) / (x2 - x1)  # Calculo de la pendiente.
    b = y1 - a * x1  # Calculo del termino independiente.

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
                ax.plot(x, (-w[1] * x - w[0]) / w[2])
        else:
            for w, name in zip(ws, labels_ws):
                ax.plot(x, (-w[1] * x - w[0]) / w[2], label=name)

    if y is not None or ws is not None:
        ax.legend()
    if title is not None:
        plt.title(title)
    plt.show()


# Apartado a)
print("Apartado a (en ventana aparte)")
x = simula_unif(50, 2, [-50, 50])
scatter(x, title="Nube de puntos uniforme")

# Apartado b)
print("Apartado b (en ventana aparte)")
x = simula_gaus(50, 2, np.array([5, 7]))
scatter(x, title="Nube de puntos gaussiana")

input("\n--- Pulsar tecla para continuar ---\n")

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
    return signo(y - a * x - b)


# CODIGO DEL ESTUDIANTE
intervalo = [-50, 50]
N = 50
a, b = simula_recta(intervalo)
x = simula_unif(N, 2, intervalo)

y = np.empty((N, ))
for i in range(N):
    y[i] = f(x[i, 0], x[i, 1], a, b)

vector_recta = np.array([b, a, -1])

scatter(x,
        y,
        ws=[vector_recta],
        labels_ws=["Frontera"],
        title="Puntos etiquetados en función de recta aleatoria")

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello

y_noise = y.copy()

# Modifica un 10% aleatorio de cada etiqueta
for label in [-1, 1]:
    y_lab = np.nonzero(y == label)[0]
    y_rand = np.random.choice(y_lab, len(y_lab) // 10, replace=False)
    y_noise[y_rand] = -y_noise[y_rand]

scatter(x,
        y_noise,
        ws=[vector_recta],
        labels_ws=["Frontera"],
        title="Puntos etiquetados con recta aleatoria (con ruido)")

# CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

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
    border_xy = (max_xy - min_xy) * 0.01

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


def getProp(datos, labels, clasificador):
    """Obtiene la proporción de puntos correctamente clasificados
    por un clasificador dado.
    Argumentos posicionales:
    - datos: datos,
    - labels: etiquetas,
    - clasificador: Clasificador"""

    # Los campos no negativos indican clasificación correcta
    signos = labels * clasificador(datos)
    return len(signos[signos >= 0]) / len(labels)


# Lista de clasificadores con su título
clasificadores = [
    (lambda x: x[:, 1] - a * x[:, 0] - b, "Recta"),
    (lambda x: (x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 1"),
    (lambda x: 0.5 * (x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 2"),
    (lambda x: 0.5 * (x[:, 0] - 10)**2 + (x[:, 1] + 20)**2 - 400, "Elipse 3"),
    (lambda x: x[:, 1] - 20 * x[:, 0]**2 - 5 * x[:, 0] + 3, "Parábola")
]

# Representa y calcula el número bien clasificado para cada tipo
for clasificador, title in clasificadores:
    plot_datos_cuad(x, y_noise, clasificador, title=title)
    print("Proporción correcta para '{}': {}".format(
        title, getProp(x, y_noise, clasificador)))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON


def ajusta_PLA(datos, label, max_iter, vini):
    """Calcula el hiperplano solución al problema de clasificación binaria.
    Argumentos posicionales:
    - datos: matriz de datos,
    - label: Etiquetas,
    - max_iter: Número máximo de iteraciones
    - vini: Valor inicial"""

    w = vini.copy()
    i = 0
    N = len(label)
    no_changes = True

    while i <= max_iter:
        # El índice actual
        j = i % N

        # Si un dato está clasificado incorrectamente
        if signo(w.dot(datos[j, :])) != label[j]:
            w = w + label[j] * datos[j]  # Actualiza el vector
            no_changes = False  # Indica que ha habido cambios

        # Si hemos pasado por todos los datos
        if j == N - 1:
            if no_changes:  # Si no ha habido cambios, para
                break
            else:  # Si los ha habido, reinicia
                no_changes = False

        i += 1

    return w


# CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0, 10):
    pass
    # CODIGO DEL ESTUDIANTE

print('Valor medio de iteraciones necesario para converger: {}'.format(
    np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

# CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT


def sgdRL(TODO):
    # CODIGO DEL ESTUDIANTE

    return w


# CODIGO DEL ESTUDIANTE
input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

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

# mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1), 1]),
        np.squeeze(x[np.where(y == -1), 2]),
        'o',
        color='red',
        label='4')
ax.plot(np.squeeze(x[np.where(y == 1), 1]),
        np.squeeze(x[np.where(y == 1), 2]),
        'o',
        color='blue',
        label='8')
ax.set(xlabel='Intensidad promedio',
       ylabel='Simetria',
       title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1), 1]),
        np.squeeze(x_test[np.where(y_test == -1), 2]),
        'o',
        color='red',
        label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1), 1]),
        np.squeeze(x_test[np.where(y_test == 1), 2]),
        'o',
        color='blue',
        label='8')
ax.set(xlabel='Intensidad promedio',
       ylabel='Simetria',
       title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# LINEAR REGRESSION FOR CLASSIFICATION

# CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

# POCKET ALGORITHM

# CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

# COTA SOBRE EL ERROR

# CODIGO DEL ESTUDIANTE

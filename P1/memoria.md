---
title: Práctica 1
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
toc-depth: 2
toc-title: Índice
---

\newpage

# 1. Búsqueda iterativa de óptimos
## 1. Implementar el algoritmo de gradiente descendente

El algoritmo de gradiente descendente está implementado en la función `gradient_descent`.
Sus argumentos posicionales son:

- `initial_point`: Punto inicial
- `fun`: Función a minimizar. Debe ser diferenciable
- `grad_fun`: Gradiente de `fun`
- `eta`: Tasa de aprendizaje
- `max_iter`: Máximo número de iteraciones

Además tenemos un argumento opcional `error2get`.
Si no pasamos este argumento hará `max_iter` iteraciones y si sí lo hacemos podrá parar antes de completar estas iteraciones si el valor de la función está por debajo de `error2get` (este criterio sólo será válido si la función tiene un mínimo con valor 0).

El código es el siguiente (omitiendo el comentario explicativo):

```python
def gradient_descent(initial_point, fun, grad_fun, eta,
                     max_iter, error2get = -math.inf):
  w = initial_point
  iterations = 0

  while fun(w) > error2get and iterations < max_iter:
    w = w - eta*grad_fun(w)
    iterations += 1

  return w, iterations
```

La variable `w` contiene la estimación actual del punto donde se alcanza el mínimo, que es inicializada con `initial_point`. `iterations` tiene el número de iteraciones.

A continuación aplicamos la fórmula del gradiente descendente (`w = w - eta*grad_fun(w)`) hasta o bien quedarse por debajo del umbral `error2get` o bien llegar al número máximo de iteraciones.

En el caso de que no pasemos umbral este tomará el valor $-\infty$, siendo entonces la comprobación del umbral siempre cierta.

## 2. Considerar la función $E(u, v) = (u^2 e^v - 2v^2e^{-u})^2$. Usar gradiente descendente para encontrar un mínimo de esta función, comenzando desde el punto $(u, v) = (1, 1)$ y usando una tasa de aprendizaje $\eta = 0.01$.

Para analizar correctamente los resultados podemos observar que $E$ es una función no negativa (por ser en cada punto el cuadrado de un número real) y que $E(0,0) = 0$, luego los mínimos globales toman valor $0$.

### a) Calcular analíticamente y mostrar la expresión del gradiente de la función $E(u, v)$

El gradiente de $E$ nos da las derivadas parciales de $E$ respecto a $u$ y $v$ en cada punto, esto es $\nabla E = (\frac{\partial E}{\partial u}, \frac{\partial E}{\partial v})$. Calculamos estas derivadas para obtener:
$$\nabla E(u,v) = (2 (u^2 e^v - 2 e^{-u} v^2) (2 e^{-u} v^2 + 2 u e^v), 2 (u^2 e^v - 4 e^{-u} v) (u^2 e^v - 2 e^{-u} v^2))$$

Su implementación por tanto queda (usando el decorador `to_numpy` para pasar a versión NumPy):

```python
@to_numpy
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
```

donde hemos utilizado la función `np.exp` que calcula la exponencial.

### b) ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de $E(u, v)$ inferior a $10^{-14}$?


Como vemos en la ejecución, el algoritmo tarda un total de **33** iteraciones en alcanzar el mínimo.
El algoritmo para por tanto no por alcanzar el número máximo de iteraciones posibles sino porque el valor del punto está por debajo de la tolerancia máxima aceptable, $10^{-14}$.

### c) ¿En qué coordenadas $(u,v)$ se alcanzó por primera vez un valor igual o menor a $10^{-14}$ en el apartado anterior?

Redondeando a 5 cifras decimales el mínimo obtenido se alcanza en el punto $(0.61921,  0.96845)$.

El valor de la función en este punto es, redondeando de nuevo a 3 cifras decimales, $5.997301\cdot 10^{-15}$.

## 3.  Considerar ahora la función $f(x, y) = x^2 + 2y^2 + 2\sin(2\pi x) \sin(2\pi y)$
### a) Usar gradiente descendente para minimizar esta función. Usar como punto inicial $(x0 = 0.1, y0 = 0.1)$, (tasa de aprendizaje $\eta = 0.01$ y un máximo de $50$ iteraciones. Generar un gráfico de cómo desciende el valor de la función con las iteraciones. Repetir el experimento pero usando $\eta = 0.1$, comentar las diferencias y su dependencia de $\eta$.

#### Cálculo del gradiente

Como en el ejercicio anterior, calculamos en primer lugar el gradiente, cuya expresión analítica es

$$\nabla f(x,y) = (2 x + 4 \pi \cos(2 \pi x) \sin(2 \pi y), 4 y + 4 \pi \sin(2 \pi x) \cos(2 \pi y))$$

De forma similar al caso del primer ejercicio, el código queda:

```python
@to_numpy
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
```

donde usamos `np.sin` y `np.cos` para calcular el seno y coseno respectivamente.


#### Código

El código que utilizamos es muy similar al de la función `gradient_descent`, salvo que guardamos en un array NumPy los valores que va tomando la función en cada punto que obtiene el método de gradiente descendente para poder hacer el posterior gráfico.

En primer lugar definimos dos arrays NumPy donde guardar los valores:

```python
resEtaPeq    = np.empty((maxIter,))
resEtaGrande = np.empty((maxIter,))
```

A continuación iteramos en los distintos valores de la tasa de aprendizaje y repetimos el método de gradiente descendente, guardando en el array `resultados` los resultados en cada iteración:

```python
for eta, resultados in [(0.01, resEtaPeq), (0.1, resEtaGrande)]:
  w = initial_point
  iterations = 0

  while iterations < maxIter:
    resultados[iterations] = f(w)
    w = w - eta*gradf(w)
    iterations += 1
```

Por último, para hacer el plot representamos ambos arrays indicando en una leyenda la tasa de aprendizaje asociada:

```python
plt.title("1.3.a: Valor de $f$ en función del número de iteraciones")
plt.plot(resEtaPeq,    'b-o', label="$\eta$ = 0.01")
plt.plot(resEtaGrande, 'k-o', label="$\eta$ = 0.1" )
plt.legend()
plt.show()
```

#### Resultados obtenidos  y valoración

El gráfico obtenido puede verse en la **Figura 1**.

![Valor de la función $f$ en $w$ en cada iteración según su tasa de aprendizaje ($\eta$).](img/1.3.a2.png){width=80%}

Como vemos en la figura, la tasa de aprendizaje afecta de forma esencial a la convergencia del método de gradiente descendente en este caso: cuando la tasa de aprendizaje es más pequeña ($\eta = 0.01$), apreciamos una rápida convergencia hacia el mínimo global de la función, y el valor va disminuyendo conforme avanza el tiempo.

En cambio, en el caso de la tasa de aprendizaje más grande ($\eta = 0.1$) el algoritmo no presenta convergencia en estas iteraciones, sino que oscila e incluso sube en algunos puntos. Esto nos indica que el algoritmo de gradiente descendente no presenta un buen comportamiento si elegimos de forma incorrecta sus parámetros.

El mejor valor obtenido se da por tanto con tasa de aprendizaje $\eta = 0.01$, con la que, como podemos ver en el gráfico de contorno (**Figura 2**), estamos cerca de un mínimo global de la función. 

![Gráfico de contorno con el mínimo obtenido. Colores cálidos indican valores mayores. La elipse nos indica la región en la que están los mínimos globales. ](img/1.3.a.png)

Del gráfico de contorno (**Figura 2**) podemos ver que es mínimo local de la región representada, pero además es global: cualquier mínimo global estará dentro de la elipse $x^2 + 2y^2 = 2$; fuera de esta $f(x,y) \geq x^2 + 2y^2 - 2 > 0$. 

Es decir, el algoritmo puede presentar buen comportamiento y llegar al mínimo global si elegimos bien los parámetros o un mal resultado si estos son incorrectos.

### b) Obtener el valor mínimo y los valores de las variables $(x,y)$ en donde se alcanzan cuando el punto de inicio se fija: $(0.1, 0.1), (1, 1),(-0.5, -0.5),(-1, -1)$. Generar una tabla con los valores obtenidos

Para este apartado utilizamos la función `gradient_descent` definida anteriormente para calcular el gradiente descendiente. Creamos un iterable de puntos iniciales `initial_points` de acuerdo a los requeridos en el enunciado.

A continuación imprimimos en formato tabla los resultados, ejecutando la función `gradient_descent` con cada punto inicial. La tasa de aprendizaje elegida es 0.01 y el número de iteraciones 50, de la misma forma que el apartado anterior.

```python
initial_points = map(np.array, [[0.1,0.1], [1,1], [-.5, -.5], [-1, -1]])

print("{:^15}  {:^15}  {:^7}".format("Inicial","Final","Valor"))
for initial in initial_points:
  w, _ = gradient_descent(initial, f, gradf, 0.01, 50)
  print("{}  {}  {: 1.3f}".format(initial, w, f(*w)))
```

La primera llamada a `print` imprime los nombres de las columnas alineados.
Los resultados pueden verse en la siguiente tabla:

| $\mathbf{(x_0,y_0)}$ | $\mathbf{(x_{\operatorname{min}},y_{\operatorname{min}})}$ | $\mathbf{E(x_{\operatorname{min}},y_{\operatorname{min}})}$ | 
|----------------------|------------------------------------------------------------|-------------------------------------------------------------|
| $(0.1, 0.1)$         | $( 0.24380, -0.23793)$                                         |   -1.82008                                                    |
| $(1, 1)$             | $( 1.21807,  0.71281)$                                         |    0.59327                                                    |
| $(-0.5, -0.5)$       | $(-0.73138, -0.23786)$                                         |   -1.33248                                                    |
| $(-1, -1)$           | $(-1.21807, -0.71281)$                                         |    0.59327                                                    |

Como vemos, incluso con la tasa de aprendizaje que da mejores resultados, esto es, 0.01, el resultado depende enormemente del punto inicial.

En primer lugar podemos observar la simetría de los resultados con los puntos iniciales $(1,1)$ y $(-1,-1)$.
Esto se debe a que la función verifica $f(x,y) = f(-x,-y)$ para cualesquiera $x,y\in \mathbb{R}$.

El mejor resultado se obtiene en el punto $(0.1,0.1)$ que, como ya comentamos en el apartado anterior,
encuentra un punto cercano a uno de los óptimos.

En el resto de casos nos quedamos en puntos muy alejados a uno de los óptimos.
Esto se debe, como podemos apreciar en el gráfico de contorno del apartado anterior, a que es una función con gran cantidad de máximos locales y por tanto el método de gradiente descendente no será capaz de escapar de estos, al menos si la tasa de aprendizaje es constante.

## 4. ¿Cuál sería su conclusión sobre la verdadera dificultad de encontrar el mínimo global de una función arbitraria?

El método de gradiente descendente es un método en principio aplicable a cualquier función diferenciable y para el que podemos asegurar de forma teórica que, bajo ciertas condiciones sobre la función como la convexidad, este método convergerá a un mínimo global de la función.
En la práctica estas condiciones no tienen por qué cumplirse en nuestros problemas (por ejemplo, las funciones $f$ y $E$ no son convexas en su dominio), por lo que no siempre obtenemos buenos resultados.

En este ejercicio vemos dos ejemplos de comportamiento muy diferentes ya que

1. en el caso de la función $E$ el algoritmo de gradiente descendente tiene una rápida convergencia hacia el mínimo global de la función, de forma robusta con independencia de cómo ajustemos los parámetros mientras que
2. en el caso de la función $f$ esta convergencia depende enormemente de la tasa de aprendizaje y del punto inicial.

La representación de ambas funciones nos ayuda a entender el por qué de este comportamiento: la función $f$ tiene muchos mínimos locales a los que converge el método de gradiente descendiente (ya que el gradiente sólo nos da información *local* sobre el comportamiento de la función) y esto provoca su mal comportamiento, mientras que la función $E$ tiene menos mínimos locales, lo que lo convierte en un problema mucho más tratable.

De esto deducimos que la verdadera dificultad reside en elegir los parámetros adecuados, cuyos valores óptimos dependerán del aspecto de la función y su cantidad de mínimos locales en los que el gradiente pueda quedarse «atrapado».

En la práctica es posible que no podamos obtener fácilmente información sobre el aspecto de las funciones que vamos a minimizar, que probablemente tengan alta dimensionalidad y no vengan dadas por una expresión analítica sencilla, por lo que tendremos que experimentar con distintas tasas de aprendizaje y puntos iniciales para ajustar el método a la función a minimizar.



\newpage

# 2. Regresión lineal
## 1. Estimación de un modelo de regresión lineal

> Estimar un modelo de regresión lineal a partir de los datos proporcionados de
> dichos números (Intensidad promedio, Simetria) usando tanto el algoritmo de la pseudo-
> inversa como Gradiente descendente estocástico (SGD). Las etiquetas serán $\{-1, 1\}$, una
> para cada vector de cada uno de los números. Pintar las soluciones obtenidas junto con los
> datos usados en el ajuste. Valorar la bondad del resultado usando $E_{\operatorname{in}}$ y $E_{\operatorname{out}}$.


### Implementación usando pseudoinversa

La implementamos en la función `pseudoinverse`, que toma dos argumentos posicionales:

- `x`: Datos en coordenadas homogéneas y
- `y`: Etiquetas asociadas (-1 o 1),

y cuyo código consiste en el cálculo de la pseudoinversa de `x` multiplicada por y a partir de la descomposición en valores singulares:

```python
def pseudoinverse(x,y):
  u, s, v = np.linalg.svd(x)
  d = np.diag([0 if np.allclose(p,0) else 1/p for p in s])
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)
```

En primer lugar hallamos la descomposición $U, S, V = \operatorname{SVD}(X)$.
A continuación, hallamos la pseudo-inversa de $S$ para lo que utilizamos la función `diag` para crear una matriz diagonal y `np.allclose` para ver si un elemento es (a efectos prácticos), cero.
Finalmente devolvemos $V^tDDVX^tY$, que nos da el cálculo mediante la pseudoinversa.

Para los datos leídos, guardamos el vector resultante en `w_pinv` (ver última sección de este apartado para la comparación de los métodos y sus resultados).

### Implementación usando SGD

La implementamos en la función `sgd`, que toma dos argumentos posicionales:

- `x`: Datos en coordenadas homogéneas y
- `y`: Etiquetas asociadas (-1 o 1)

y los siguientes argumentos opcionales:

- `eta`: Tasa de aprendizaje,
- `max_iter`: máximo número de iteraciones y
- `batch_size`: tamaño del *batch*.

En primer lugar inicializamos el vector de solución actual `w`, el número de iteraciones `iterations`, un vector de índices `idxs` y un entero que indica a partir de qué punto tomamos índices para tomar la muestra.

```python
w = np.zeros((3,))
iterations = 0
idxs = np.arange(len(x)) # vector de índices
batch_start = 0 # Comienzo de la muestra
```
A continuación, en un bucle hasta que `iterations = max_iter` tomamos los siguientes pasos:

- Si `batch_start` es 0, hacemos una permutación de los índices: `idxs = np.random.permutation(idxs)`.
- Tomamos un *slice* de tamaño `batch_size` (salvo posiblemente en la última iteración de una época), empezando por `batch_start`: `idx = idxs[batch_start: batch_start + batch_size]`
- Actualizamos el vector `w` y el número de iteraciones:     `w = w - eta*dErr(x[idx,:],y[idx],w)` y `iterations += 1`
- Finalmente aumentamos `batch_start` y, si hemos llegado al final, reiniciamos:
```python
batch_start += batch_size
if batch_start > len(x):
   batch_start = 0
```

El código completo de la función (omitiendo el *docstring*) es:

```python
def sgd(x, y, eta = 0.01, max_iter = 1000, batch_size = 32):

  w = np.zeros((3,))
  iterations = 0

  idxs = np.arange(len(x)) # vector de índices
  batch_start = 0 # Comienzo de la muestra

  while iterations < max_iter:
    if batch_start == 0: # Si empezamos una época, shuffle
      idxs = np.random.permutation(idxs)
    # Toma índices
    idx = idxs[batch_start: batch_start + batch_size]

    w = w - eta*dErr(x[idx,:],y[idx],w)
    iterations += 1

    # Actualiza el comienzo del batch
    batch_start += batch_size
    if batch_start > len(x): # Si hemos llegado al final reinicia
      batch_start = 0
  return w
```

Para los datos leídos, guardamos el vector resultante en `w_sgd` (ver última sección de este apartado para la comparación de los métodos y sus resultados).

### Pintar las soluciones obtenidas

Para la representación en este y el siguiente apartado creamos la función `scatter`, que nos permite representar en un *scatter plot* los puntos así como el modelo estimado.


La función scatter puede tomar entre uno y cuatro parámetros:

- `x` el vector de puntos a representar (en coordenadas homogéneas, esto es, añadiendo un $1$ a cada punto),
- `y` el vector de clases (1 o -1),
- `ws` un iterable de vectores que representan rectas y
- `labels_ws`, las etiquetas de estas rectas.

Inicialmente fijamos los límites del gráfico:

```python
_, ax = plt.subplots()
xmin, xmax = np.min(x[:,1]), np.max(x[:,1])
ax.set_xlim(xmin, xmax)
ax.set_ylim(np.min(x[:,2]),np.max(x[:,2]))
```

A continuación, si no hay clases mostramos simplemente los puntos (`ax.scatter(x[:,1], x[:,2])`) y en otro caso los mostramos siguiendo el código que implementé para la práctica 0:

```python
class_colors = {-1 : 'green', 1 : 'blue'}
# Para cada clase:
for cls, name in [(-1,"Clase -1"), (1,"Clase 1")]:
  # Obten los miembros de la clase
  class_members = x[y == cls]
  
  # Representa en scatter plot
  ax.scatter(class_members[:,1],
             class_members[:,2],
             c = class_colors[cls],
             label = name)
```

Además, en caso de que haya rectas a representar las mostramos (con o sin etiqueta dependiendo de si se ha pasado ese argumento):

```python
x = np.array([xmin,xmax])
if labels_ws is None:
  for w in ws:
    ax.plot(x,(-w[1]*x - w[0])/w[2])
else:
  for w, name in zip(ws, labels_ws):
    ax.plot(x,(-w[1]*x - w[0])/w[2], label = name)
```

Finalmente mostramos en caso de que sea necesario la leyenda utilizando `ax.legend()`.


****

En el caso de este ejercicio mostramos el *scatter plot* con la siguiente llamada `scatter(x,y, [w_sgd, w_pinv], ["SGD", "Pinv"])`.

El resultado puede verse en la **Figura 3**.

![Rectas obtenidas por regresión lineal para el dataset de números junto con los datos.](img/2.1.png){width=75%}

### Valorar la bondad del resultado

Los errores obtenidos pueden verse en la siguiente tabla:

|                          | SGD     | Pinv    |
|--------------------------|---------|---------|
| $E_{\operatorname{in}}$  | 0.07968 | 0.07919 |
| $E_{\operatorname{out}}$ | 0.13221 | 0.13095 |

Como vemos la pseudo-inversa obtiene ligeramente mejores resultados, hecho también apreciable en la **Figura 3**.

La métrica $E_{\operatorname{out}}$ obtiene un error mayor que $E_{\operatorname{in}}$, lo que es esperable ya que el algoritmo no puede ajustarse a los datos de test, que no se le proporcionan durante la ejecución.

Estos resultados nos indican que, para este problema, la pseudoinversa nos proporcionará mejores resultados, ya que resuelve de forma exacta las ecuaciones que minimizan el error $E_{\operatorname{in}}$ (lo que, bajo condiciones usuales, minimizará también $E_{\operatorname{out}}$).

Sin embargo, existen situaciones en las que puede ser preferible utilizar el método de gradiente descendiente estocástico ya que tiene un menor tiempo de ejecución y su componente aleatorio puede permitirnos salir de mínimos locales en funciones no convexas en la práctica.

## 2. Realizar el siguiente experimento

> En este apartado exploramos como se transforman los errores $E_{\operatorname{in}}$ y 
> $E_{\operatorname{out}}$ cuando aumentamos la complejidad del modelo lineal usado. 
> Ahora hacemos uso de la función `simula_unif (N, 2, size)` que nos 
> devuelve `N` coordenadas 2D de puntos uniformemente muestreados dentro del
> cuadrado definido por $[-\texttt{size}, \texttt{size}] \times [-\texttt{size}, \texttt{size}]$.

### a) Generar una muestra de entrenamiento de $N = 1000$ puntos en el cuadrado $X = [-1, 1] \times [-1, 1]$. Pintar el mapa de puntos 2D.

Para ello utilizamos la función `simula_unif` del template, con `N = 1000` muestras, en `d = 2` dimensiones con coordenadas acotadas en valor absoluto por `size = 1`:

```python
x = simula_unif(1000, 2, 1)
```

Lo hacemos dentro de la función `genera_datos` (que contiene también la implementación del apartado b).
Para la representación gráfica utilizamos la función `scatter` descrita en el ejercicio anterior, llamada de la forma `scatter(x)` (ver apartado c para el código completo).

Los datos se muestran en la **Figura 4**.

![Datos aleatorios generador en el primer apartado del experimento](img/2.2.a.png)


### b)  Consideremos la función $f(x_1 , x_2) = \operatorname{sign}((x_1 - 0.2)^2 + x_2^2 - 0.6)$ que usaremos para asignar una etiqueta a cada punto de la muestra anterior. Introducimos ruido sobre las etiquetas cambiando aleatoriamente el signo de un 10 % de las mismas. Pintar el mapa de etiquetas obtenido.

Definimos la función `f_label` que asigna la etiqueta a un array NumPy:

```python
@to_numpy
def f_label(x1,x2):
  """Función de apartado 2.2.b"""
  return np.sign((x1 -0.2)**2 + x2**2 - 0.6)
```

Asignamos al 90% inicial de los datos las etiquetas con esta función y al 10% de forma aleatoria en la función `genera_datos`, que también hace el apartado anterior:


```python
def genera_datos():
  """Genera datos aleatorios para el experimento."""

  x = simula_unif(1000, 2, 1)


  y_f    = f_label(x[:900,:].T)
  y_rand = np.random.choice([-1,1], 100)
  y      = np.hstack((y_f, y_rand))
  
  return x,y
```

Finalmente juntamos las etiquetas en un vector `y` utilizando la función `hstack` que une ambos vectores.
Las mostramos con `scatter(x,y)` (ver apartado c para código completo).

Los datos etiquetados se muestran en la **Figura 5**.

![Datos etiquetados de acuerdo al procedimiento descrito en el apartado b) del experimento](img/2.2.b.png)


### c) Usando como vector de características $(1, x_1 , x_2)$ ajustar un modelo de regresión lineal al conjunto de datos generado y estimar los pesos w. Estimar el error de ajuste $E_{\operatorname{in}}$ usando Gradiente Descendente Estocástico (SGD).

Para preparar el vector de características, apoyándonos en la función anterior, definimos la función `genera_hom` que genera los vectores de características (es decir, que transforma a coordenadas homogéneas los puntos):

```python
def genera_hom():
  """Genera los datos en coordenadas homogéneas"""
  x,y = genera_datos()
  x_hom = np.hstack((np.ones((1000,1)), x))
  return x_hom, y
```

El siguiente código genera los datos y los muestra (apartado a), genera sus etiquetas y las muestra (apartado b) y finalmente calcula el vector `w` con el método de gradiente descendente estocástico `sgd` definido para el ejercicio anterior y estima el error (apartado c):

```python
x,y = genera_hom()
scatter(x)
scatter(x,y)

w = sgd(x,y)
print("Bondad del resultado del experimento (1 ejecución)")
print("  Ein:  {}".format(Err(x,y,w)))
```

El error obtenido en la iteración con los datos de la figura del apartado anterior es de $0.91701$, un error muy cercano al máximo error posible, 1 (ver último apartado para análisis de estos resultados).

### d) Ejecutar el experimento 1000 veces

> Ejecutar todo el experimento definido por (a)-(c) 1000 veces (generamos 1000
> muestras diferentes) y
> 
> - Calcular el valor medio de los errores Ein de las 1000 muestras.
> - Generar 1000 puntos nuevos por cada iteración y calcular con ellos el valor
> de Eout en dicha iteración. Calcular el valor medio de Eout en todas las
> iteraciones.

El experimento se encapsula en la función `experimento`, que hace el procedimiento del apartado anterior sin mostrar los gráficos.

```python
def experimento():
  x,y = genera_hom()
  w = sgd(x, y)
  Ein  = Err(x,y,w)
  
  x_test, y_test = genera_hom()
  Eout = Err(x_test, y_test, w)
  return np.array([Ein, Eout])
```

Las primeras 3 líneas hacen el experimento, mientras que las últimas líneas crean 1000 datos nuevos bajo las mismas condiciones (`x_test` e `y_test`) y calculamos el error `Eout`.

Finalmente calculamos el error medio y lo mostramos:

```python
Nexp = 1000
errs = 0
for _ in range(Nexp):
   errs = errs + experimento()
Ein_medio, Eout_medio = errs/Nexp

print("Bondad del resultado del experimento ({} ejecuciones)".format(Nexp))
print("  Ein:  {}".format(Ein_medio))
print("  Eout: {}".format(Eout_medio))
```

### e) Valore que tan bueno considera que es el ajuste con este modelo lineal a la vista de los valores medios obtenidos de $E_{\operatorname{in}}$ y $E_{\operatorname{out}}$.

Los resultados obtenidos en la ejecución de ejemplo son un error medio $E_{\operatorname{in}}$ de $0.90789$ y $E_{\operatorname{out}}$ de $0.91389$.

Como vemos los errores de este modelo, tanto `Ein` como `Eout` son muy cercanos a uno, por lo que el modelo realiza un muy mal ajuste.

Esto se debe a que los datos no siguen una distribución lineal: la función óptima sería la función $g(x_1, x_2) = (x_1 - 0.2)^2 + x_2^2 - 0.6$, que viene dada por una fórmula cuadrática.
No existe ninguna recta que pueda aproximar esta circunferencia de forma adecuada, por lo que el error es muy alto.

Una posible solución sería hacer un ajuste lineal utilizando como vector de características $(1,x_1, x_2, x_1^2, x_2^2)$, que nos permitiría un ajuste cercano a $g$.


# Bonus

## 1. **Método de Newton**. Implementar el algoritmo de minimización de Newton y aplicarlo a la función $f(x, y)$ dada en el ejercicio 3. Desarrolle los mismos experimentos usando los mismos puntos de inicio.

### Implementación del algoritmo

Implementamos el método en la función `newton`:

```python
def newton(initial_point, fun, grad_fun, hessian, eta, max_iter):
  w = initial_point
  w_list = [initial_point]
  iterations = 0

  while iterations < max_iter:
    w = w - eta*np.linalg.inv(hessian(w)).dot(grad_fun(w))
    w_list.append(w)
    iterations += 1

  return np.array(w_list)
```

El código es similar al método de gradiente descendente. 
Toma un argumento adicional `hessian` que se corresponde con la función que asigna a cada punto la matriz hessiana de `fun` en ese punto.

La ecuación que actualiza el punto en la iteración actual se actualiza haciendo el producto matricial con la inversa de la hessiana (haciendo uso de `np.linalg.inv`): 
```python
w = w - eta*np.linalg.inv(hessian(w)).dot(grad_fun(w))
```

Además, para ayudar a la representación en este caso añadimos una lista `w_list` que guarda el punto en cada iteración.

El método de Newton se utiliza normalmente para hallar ceros de una función.
En este caso lo utilizamos para encontrar los ceros del gradiente de $f$.

Para aplicarla a la función $f$ necesitaremos la hessiana de $f$, es decir, la matriz de las segundas derivadas de $f$:

```python
@to_numpy
def hessianf(x,y):
  return np.array(
    [2 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
     8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
     8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
     4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    ]).reshape((2,2))
```

### Generar un gráfico de cómo desciende el valor de la función

Para representar el gráfico simplemente utilizamos la función para hallar el array de puntos y los evaluamos con `f` mediante la función `apply_along_axis`:

```python
resEtaPeqNewton = np.apply_along_axis(f, 1, 
  newton(np.array([0.1,0.1]), f, gradf, hessianf,  0.01, 50))
resEtaGrandeNewton = np.apply_along_axis(f, 1, 
  newton(np.array([0.1,0.1]), f, gradf, hessianf,  0.1, 50))

plt.plot(resEtaPeq,    'b-o', label="GD, $\eta$ = 0.01")
plt.plot(resEtaGrande, 'k-o', label="GD, $\eta$ = 0.1" )
plt.plot(resEtaPeqNewton,    'g-o', label="Newton, $\eta$ = 0.01")
plt.plot(resEtaGrandeNewton, 'c-o', label="Newton, $\eta$ = 0.1" )
plt.legend()
plt.show()
```

El gráfico puede verse en la **Figura 6**.

![Curva de decrecimiento de método de gradiente descendente y método de Newton para distintos valores de tasa de aprendizaje](img/bonus.1.png)

Para su análisis ver el apartado siguiente.

### Extraer conclusiones sobre las conductas de los algoritmos comparando la curva de decrecimiento de la función calculada en el apartado anterior y la correspondiente obtenida con gradiente descendente.

Como vemos el algoritmo de Newton no consigue minimizar la función correctamente, tampoco en el punto de inicio que da buenos resultados con el gradiente descendente e independientemente de la tasa de aprendizaje escogida. Esto se debe a que, como estamos encontrando puntos en los que el gradiente se anula, encontramos puntos de silla que no son mínimos de la función.

La siguiente tabla muestra los resultados obtenidos para distintos puntos de inicio.

|$w_{0}$  | $w_{\text{fin}}$ | $f(w_{\text{fin}})$| $\nabla f(w_{\text{fin}})$|
|------------------|------------------|--------------------|---------------------------|
|$(-0.5, -0.5)$ | $( 0.00002,  0.00002)$ |  0.72548 | $(-0.00032, -0.00064)$ |
|  $(0.1, 0.1)$ | $(-0.94915, -0.97458)$ | 0.00000 | $( 0.00198,  0.00204)$ |
|       $(1,1)$ | $(-0.47512, -0.48781)$ |2.90040 | $( 0.00064,  0.00129)$ |
|   $( -1, -1)$ | $( 0.94915,  0.97458)$ | 2.90040 | $(-0.00064, -0.00129)$ |

El código de generación de la tabla es un simple bucle:

```python
initial_points = zip(["0.1, 0.1", "1,1", "-0.5, -0.5", "-1, -1"],
                     map(np.array, [[0.1,0.1], [1,1], [-.5, -.5], [-1, -1]]))
print("Punto inicial, punto final, valor final, gradiente final:")
steps_for = {}
for label, initial in initial_points:
  points = newton(initial, f, gradf, hessianf,  0.01, 800)
  steps_for[label] = points
  print("{:>10}".format(label), points[-1,:], 
        "{: 1.5f}".format(f(points[-1,:])), gradf(points[-1,:]))
```

Es decir, el algoritmo está minimizando correctamente el gradiente pero no la función.
Como conclusión vemos que, aunque el método de Newton puede dar buenos resultados en otras situaciones, no es capaz de hacerlo cuando la función tiene muchos puntos de silla.

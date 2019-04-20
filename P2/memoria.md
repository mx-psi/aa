---
title: Práctica 2
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
colorlinks: true
toc-depth: 2
toc-title: Índice
header-includes:
- \newcommand{\x}{\mathbf{x}}
- \newcommand{\w}{\mathbf{w}}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
---

\newpage

# 1. Ejercicio sobre la complejidad de $\mathcal{H}$ y el ruido

> En este ejercicio debemos aprender la dificultad que introduce la aparición de ruido en las
> etiquetas a la hora de elegir la clase de funciones más adecuada. Haremos uso de tres funciones
> ya programadas:
>
> - `simula_unif(N, dim, rango)`, que calcula una lista de `N` vectores de dimensión `dim`. 
>    Cada vector contiene `dim` números aleatorios uniformes en el intervalo `rango`.
> - `simula_gaus(N, dim, sigma)`, que calcula una lista de `N` de vectores de dimensión `dim`,
>    donde cada posición del vector contiene un número aleatorio extraido de una distribución 
>    Gaussiana de media 0 y varianza dada, para cada dimension, por la posición del vector `sigma`.
> - `simula_recta(intervalo)`, que simula de forma aleatoria los parámetros, $v = (a, b)$ de una
>    recta, $y = ax + b$, que corta al cuadrado $[-50, 50] \times [-50, 50]$.

## 1. Dibujar una gráfica con la nube de puntos de salida correspondiente.

Para este ejercicio utilizo la función `scatter` que implementé en la práctica anterior.
Su funcionamiento se explica en la sección [Apéndice: `scatter`].

### a) Considere $N = 50$, $\operatorname{dim} = 2$, $\operatorname{rango} = [-50, +50]$ con `simula_unif(N, dim, rango)`.

En primer lugar simulamos los datos con `simula_unif`,
```python
x = simula_unif(50, 2, [-50, 50])
```

A continuación mostramos el gráfico con la función `scatter`,
```python
scatter(x, title = "Nube de puntos uniforme")
```

El resultado puede verse en la figura 1.

![Imagen generada en el apartado 1.1.a](img/1.1.a.png){width=70%}

Como vemos los datos se reparten de forma uniforme en la figura


### b) Considere $N = 50$, $\operatorname{dim} = 2$ y $\sigma = [5, 7]$ con `simula_gaus(N, dim, sigma)`

En primer lugar simulamos los datos con `simula_gaus`,
```python
x = simula_gaus(50, 2, [-50, 50])
```

A continuación mostramos el gráfico con la función `scatter`,
```python
scatter(x, title = "Nube de puntos gaussiana")
```

El resultado puede verse en la figura 2.

![Imagen generada en el apartado 1.1.b](img/1.1.b.png){width=70%}

Como vemos los datos se concentran alrededor de la media de la distribución gaussiana.

## 2. Generar una muestra de puntos 2D con etiquetas según el lado al que queden de una recta

> Con ayuda de la función `simula_unif` generar una muestra de puntos 2D a los que vamos añadir una etiqueta 
> usando el signo de la función $f(x, y) = y - ax - b$, es decir el signo de la distancia de cada punto a la recta 
> simulada con `simula_recta`.

Fijamos como número de puntos `N = 100` y como intervalo `[-50,50]`.

En primer lugar generamos una recta aleatoria,
```python
intervalo = [-50, 50]
a, b = simula_recta(intervalo
vector_recta = np.array([b, a, -1])
```
e indicamos los coeficientes de la recta expresada como hiperplano para la representación.

A continuación generamos los datos aleatorios de forma uniforme,
```python
N = 50
x = simula_unif(N, 2, intervalo)
```

Por último asignamos las etiquetas a partir de la función `f` ya provista en la plantilla,
```python
y = np.empty((N, ))
for i in range(N):
  y[i] = f(x[i, 0], x[i, 1], a, b)
```

### a) Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello. (Observe que todos los puntos están bien clasificados respecto de la recta)


Para la representación utilizamos la función `scatter`, apoyándonos en el vector de la recta `vector_recta` calculado anteriormente,
```python
scatter(
  x,
  y,
  ws = [vector_recta],
  labels_ws = ["Frontera"],
  title = "Puntos etiquetados en función de recta aleatoria")
```

El resultado puede verse en la figura 3.
Como vemos todos los puntos están bien clasificados; los que quedan por encima de la recta tienen una etiqueta mientras que el resto tienen la etiqueta contraria.

![Puntos clasificados respecto a una recta aleatoria](img/1.2.a.png){width=80%}

### b) Modifique de forma aleatoria un 10 % etiquetas positivas y otro 10 % de negativas y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior. (Ahora hay puntos mal clasificados respecto de la recta)

Creamos un vector `y_noise = y.copy()` que es copia de las etiquetas correctas, en el que modificaremos aleatoriamente algunas etiquetas.

Para cada etiqueta `label` de entre $\{-1,1\}$ tomamos el conjunto de índices del array original `y` donde tenemos esta etiqueta,
```python
y_lab = np.nonzero(y == label)[0]
```
A continuación tomamos un 10% aleatorio de entre estos (redondeando hacia arriba), haciendo uso de la función `np.random.choice`,
```python3
y_rand = np.random.choice(y_lab, math.ceil(0.1*len(y_lab)), replace = False)
```
y cambiamos el signo de las etiquetas en estos índices,
```python
y_noise[y_rand] = -y_noise[y_rand]
```

El código completo queda
```python
for label in {-1, 1}:
  y_lab = np.nonzero(y == label)[0]
  y_rand = np.random.choice(y_lab, math.ceil(0.1*len(y_lab)), replace = False)
  y_noise[y_rand] = -y_noise[y_rand]
```

Finalmente hacemos la representación con las nuevas etiquetas con `scatter`
```python
scatter(
  x,
  y_noise,
  ws = [vector_recta],
  labels_ws = ["Frontera"],
  title = "Puntos etiquetados con recta aleatoria (con ruido)")
```

El resultado puede verse en la figura 4.
Como vemos en este caso hay algunos puntos, tanto en la parte superior como en la parte inferior de la recta que no están clasificados correctamente.

![Puntos clasificados respecto a una recta aleatoria con ruido](img/1.2.b.png){width=80%}

## 3. Comparación de clasificadores

> Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
> 
> 1. $f (x, y) = (x - 10)^2 + (y - 20)^2 - 400$
> 2. $f (x, y) = 0.5(x + 10)^2 + (y - 20)^2 - 400$
> 3. $f (x, y) = 0.5(x - 10)^2 - (y + 20)^2 - 400$
> 4. $f (x, y) = y - 20x^2 - 5x + 3$
>
> Visualizar el etiquetado generado en 2b junto con cada una de las gráficas de cada una de
> las funciones. Comparar las formas de las regiones positivas y negativas de estas nuevas
> funciones con las obtenidas en el caso de la recta ¿Son estas funciones más complejas
> mejores clasificadores que la función lineal? ¿En que ganan a la función lineal? Explicar el
> razonamiento.

Nos apoyamos en la función `plot_datos_cuad` proporcionada en la plantilla.
Además definimos la función `getPorc`, que obtiene el porcentaje de `datos` correctamente clasificados (de acuerdo a `labels`) por un `clasificador`. El cuerpo de la función es (omitiendo comentarios):
```python
def getPorc(datos, labels, clasificador):
  signos = labels*clasificador(datos)
  return 100*len(signos[signos >= 0])/len(labels)
```

Es decir, calculamos el producto para cada dato de su etiqueta por la salida del clasificador.
Si el resultado es positivo tendremos que la clasificación ha sido correcta y será incorrecta en otro caso.
Devolvemos el porcentaje de `signos` no negativos de entre los totales.

Además, creamos una lista de pares de clasificadores (dados por funciones anónimas) y sus nombres,
```python
clasificadores = [
  (lambda x: x[:, 1] - a*x[:, 0] - b, "Recta"),
  (lambda x: (x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 1"),
  (lambda x: 0.5*(x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400, "Elipse 2"),
  (lambda x: 0.5*(x[:, 0] - 10)**2 + (x[:, 1] + 20)**2 - 400, "Elipse 3"),
  (lambda x: x[:, 1] - 20*x[:, 0]**2 - 5*x[:, 0] + 3, "Parábola")
]
```

Para generar la representación y los datos bien clasificados obtenemos, para cada clasificador `f`, su representación con `plot_datos_cuad` y su porcentaje de correctos con `getPorc`:
```python
for f, title in clasificadores:
  plot_datos_cuad(x, y_noise, f, title = title)
  print("Correctos para '{}': {}".format(title, getPorc(x, y_noise, f)))
```

Los resultados de porcentaje de acierto pueden verse en la siguiente tabla:

| Clasificador | % de acierto |
|--------------|--------------|
| Recta        |   92%        |
| Elipse 1     |   26%        |
| Elipse 2     |   26%        |
| Elipse 3     |   40%        |
| Parábola     |   76%        |

Los gráficos pueden verse en las figuras.



# 2. Modelos lineales
## 1. Algoritmo Perceptron

> Implementar la función `ajusta_PLA(datos, label, max_iter, vini)`
> que calcula el hiperplano solución a un problema de clasificación binaria usando el algoritmo
> PLA. La entrada `datos` es una matriz donde cada item con su etiqueta está representado
> por una fila de la matriz, `label` el vector de etiquetas (cada etiqueta es un valor +1 o -1),
> `max_iter` es el número máximo de iteraciones permitidas y `vini` el valor inicial del vector.
> La función devuelve los coeficientes del hiperplano.

### a) Ejecutar el algoritmo PLA con los datos simulados en los apartados 2a de la sección.1.

> Inicializar el algoritmo con: a) el vector cero y, b) con vectores de números aleatorios
> en [0, 1] (10 veces). Anotar el número medio de iteraciones necesarias en ambos para
> converger. Valorar el resultado relacionando el punto de inicio con el número de
>iteraciones.

#### Inicializado con el vector cero

#### Inicializado con vectores de números aleatorios

#### Valoración del resultado

### b) Hacer lo mismo que antes usando ahora los datos del apartado 2b de la sección.1. ¿Observa algún comportamiento diferente? En caso afirmativo diga cual y las razones para que ello ocurra.

## 2. Regresión logística

> En este ejercicio crearemos nuestra propia función
> objetivo $f$ (una probabilidad en este caso) y nuestro conjunto de datos $\mathcal{D}$ 
> para ver cómo funciona regresión logística. Supondremos por simplicidad que $f$ es una probabilidad con
> valores $\{0,1\}$ y por tanto que la etiqueta $y$ es una función determinista de $\x$.
> Consideremos $d = 2$ para que los datos sean visualizables, y sea $\mathcal{X} = [0, 2] \times [0, 2]$
> con probabilidad uniforme de elegir cada $\x \in \mathcal{X}$ . 
> Elegir una línea en el plano que pase por $\mathcal{X}$ como la frontera entre $f(\x) = 1$ 
> (donde $y$ toma valores $+1$) 
> y $f(\x) = 0$ (donde y toma valores $-1$), para ello seleccionar dos puntos aleatorios del plano 
> y calcular la línea que pasa por ambos. 
> Seleccionar $N = 100$ puntos aleatorios $\{\x_n\}$ de $\mathcal{X}$ y evaluar las respuestas
> $\{y_n\}$ de todos ellos respecto de la frontera elegida.


En primer lugar generams los datos de forma similar al ejercicio 1.
Generamos la recta,
```python
intervalo = [-2, 2]
a, b = simula_recta(intervalo)
```
los datos (con su versión homogénea),
```python
N = 100
datos = simula_unif(N, 2, intervalo)
datos_hom = np.hstack((np.ones((N, 1)), datos))
```
y por último las etiquetas,
```python
labels = np.empty((N, ))
for i in range(N):
  labels[i] = f(x[i, 0], x[i, 1], a, b)
```

### a)  Implementar Regresión Logística (RL) con Gradiente Descendente Estocástico (SGD) 

>  Implementar Regresión Logística(RL) con Gradiente Descendente Estocástico (SGD)
>  bajo las siguientes condiciones:
> 
> - Inicializar el vector de pesos con valores 0.
> - Parar el algoritmo cuando $\lVert \w^{(t-1)} - \w^{(t)}\rVert < 0.01$, donde $\w(t)$ denota el vector
>   de pesos al final de la época $t$. Una época es un pase completo a través de los $N$
>   datos.
> - Aplicar una permutación aleatoria, $1, 2, \dots, N$ , en el orden de los datos antes de
>   usarlos en cada época del algoritmo.
> - Usar una tasa de aprendizaje de $\eta = 0.01$

El algoritmo se implementa en la función `sgdRL`, que toma como argumentos los `datos` y las etiquetas (`labels`).
Además toma como argumento la tasa de aprendizaje `eta` con valor por defecto `0.01`.

Implementamos además una función de gradiente, `gradRL`, con el gradiente para un punto,
```python
def gradRL(dato, label, w):
  return -label*dato/(1 + np.exp(label*w.dot(dato)))
```
Utilizamos un tamaño de batch 1.


En primer lugar inicializamos el vector de pesos `w`, una variable que nos dice si ha habido cambios suficientemente grandes del vector de pesos en esta iteración (`ha_cambiado`) y un vector de índices `idxs`,
```python
  N, dim = datos.shape
  w = np.zeros(dim)
  ha_cambiado = True  # Si ha variado en la época actual
  idxs = np.arange(N)  # vector de índices
```

El bucle principal del algoritmo ejecuta una época cada vez. 
En primer lugar copia el vector de pesos para comprobar si cambia en la época actual,
a continuación hace una permutación del vector de índices y finalmente aplica el algoritmo de gradiente descendente para cada dato:
```python
while ha_cambiado:
  w_old = w.copy()
  idxs = np.random.permutation(idxs)
  for idx in idxs:
    w += -eta*gradRL(datos[idx], labels[idx], w)
  ha_cambiado = np.linalg.norm(w - w_old) > 0.01
```
La comprobación final nos indica si ha cambiado el vector de pesos en esta iteración.

Finalmente devolvemos `w`. El código completo de la función (omitiendo comentarios) es:
```python
def sgdRL(datos, labels, eta = 0.01):
  N, dim = datos.shape
  w = np.zeros(dim)
  ha_cambiado = True
  idxs = np.arange(N)

  while ha_cambiado:
    w_old = w.copy()
    idxs = np.random.permutation(idxs)
    for idx in idxs:
      w += -eta*gradRL(datos[idx], labels[idx], w)
    ha_cambiado = np.linalg.norm(w - w_old) > 0.01

  return w
```

### b) Usar la muestra de datos etiquetada para encontrar nuestra solución $g$ y estimar $E_{\operatorname{out}}$ usando para ello un número suficientemente grande de nuevas muestras ($>999$).


# 3. Bonus

## Clasificación de dígitos

> (1.5 puntos) Clasificación de Dígitos. Considerar el conjunto de datos de los dígitos manuscritos 
> y seleccionar las muestras de los dígitos 4 y 8. Usar los ficheros de entrenamiento (training)
> y test que se proporcionan. Extraer las características de **intensidad promedio** y **simetría** en
> la manera que se indicó en el ejercicio 3 del trabajo 1.

### 1. Plantear un problema de clasificación binaria que considere el conjunto de entrenamiento como datos de entrada para aprender la función $g$.

### 2. Usar un modelo de Regresión Lineal y aplicar PLA-Pocket como mejora. Responder a las siguientes cuestiones.

#### a) Generar gráficos separados (en color) de los datos de entrenamiento y test junto con la función estimada.

#### b) Calcular $E_{\operatorname{in}}$ y $E_{\operatorname{test}}$ (error sobre los datos de test).

#### c) Obtener cotas sobre el verdadero valor de $E_{\operatorname{out}}$ . Pueden calcularse dos cotas una basada en $E_{\operatorname{in}}$ y otra basada en $E_{\operatorname{test}}$ . Usar una tolerancia $\delta= 0.05$. ¿Que cota es mejor?


# Apéndice: `scatter`

**Nota:** Esta sección incluye la explicación de la función `scatter`, que implementé para la práctica 1.
No incluye ninguna información nueva respecto de lo comentado en la memoria de esa práctica.

***

La función scatter puede tomar entre uno y cuatro parámetros:

- `x` el vector de puntos a representar (en coordenadas homogéneas, esto es, añadiendo un $1$ a cada punto),
- `y` el vector de clases (1 o -1),
- `ws` un iterable de vectores que representan rectas y
- `labels_ws`, las etiquetas de estas rectas.

Además podemos pasarle como argumento opcional un título mediante el parámetro opcional `title`.

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



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

\newpage

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

### Implementación

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

### Resultados y análisis

Los resultados de porcentaje de acierto pueden verse en la siguiente tabla:

| Clasificador | % de acierto |
|--------------|--------------|
| Recta        |   92%        |
| Elipse 1     |   26%        |
| Elipse 2     |   26%        |
| Elipse 3     |   40%        |
| Parábola     |   76%        |

Los gráficos pueden verse en las figuras 5,6,7,8 y 9.
Como vemos, aunque los clasificadores como elipses o parábolas son más complejos,
no son en este caso más precisos en su clasificación, ya que no se ajustan a la distribución de los datos.
Concluimos por tanto que debemos escoger un clasificador que se ajuste a esta distribución sin llegar al sobreajuste.

![Clasificador 1 (recta)](img/1.3.1.png){width=82%}

![Clasificador 2 (elipse)](img/1.3.2.png){width=82%}

![Clasificador 3 (elipse)](img/1.3.3.png){width=82%}

![Clasificador 4 (elipse)](img/1.3.4.png){width=82%}

![Clasificador 5 (parábola)](img/1.3.5.png){width=82%}

\newpage

# 2. Modelos lineales
## 1. Algoritmo Perceptron

> Implementar la función `ajusta_PLA(datos, label, max_iter, vini)`
> que calcula el hiperplano solución a un problema de clasificación binaria usando el algoritmo
> PLA. La entrada `datos` es una matriz donde cada item con su etiqueta está representado
> por una fila de la matriz, `label` el vector de etiquetas (cada etiqueta es un valor +1 o -1),
> `max_iter` es el número máximo de iteraciones permitidas y `vini` el valor inicial del vector.
> La función devuelve los coeficientes del hiperplano.

La función `ajusta_PLA` implementa el algoritmo PLA.
La variable `w` guarda el vector de pesos actual e `it` guarda el número de iteraciones actual.

En primer lugar inicializamos `w` con una copia del vector de inicio `vini`,
```python
w = vini.copy()
```

A continuación, para cada época (hasta un máximo de `max_iters` épocas), copiamos `w_old = w.copy()` y actualizamos el vector de pesos si no predice el signo correcto para un dato concreto,
```python
for dato, label in zip(datos, labels):
  if signo(w.dot(dato)) != label:
    w += label*dato
```

Por último, si no ha habido cambios (esto es, si el vector `w` es el mismo que `w_old`), devuelve el vector de pesos y el número de iteraciones,
```python
if np.all(w == w_old): # No hay cambios
  return w, it
```

El algoritmo completo queda
```python
def ajusta_PLA(datos, labels, max_iter, vini):
  w = vini.copy()
  
  for it in range(1, max_iter + 1):
    w_old = w.copy()
    
    for dato, label in zip(datos, labels):
      if signo(w.dot(dato)) != label:
        w += label*dato
    
    if np.all(w == w_old): # No hay cambios
      return w, it
  
  return w, it
```

### a) Ejecutar el algoritmo PLA con los datos simulados en los apartados 2a de la sección.1.

> Inicializar el algoritmo con: a) el vector cero y, b) con vectores de números aleatorios
> en [0, 1] (10 veces). Anotar el número medio de iteraciones necesarias en ambos para
> converger. Valorar el resultado relacionando el punto de inicio con el número de
>iteraciones.

Para utilizarla tanto en este apartado como en el siguiente, definimos el clasificador asociado al vector normal a un hiperplano,
```python
def clasifHiperplano(w):
  return lambda x: x.dot(w)
```

Además, para reutilizar el código en el apartado b, definimos una función que ejecuta el algoritmo PLA en las condiciones descritas por el enunciado. La cabecera será `testPLA(x, y, max_iters = 1000)`.

#### Inicializado con el vector cero

En primer lugar ejecutamos con el vector cero,
```python
w, its = ajusta_PLA(x, y, max_iters, np.zeros(3))
  
print("Vector inicial cero")
print("Iteraciones: {} épocas".format(its))
print("% correctos: {}%".format(getPorc(x, y, clasifHiperplano(w))))
```
mostramos tanto el porcentaje correcto como el número de épocas necesarias para converger.

#### Inicializado con vectores de números aleatorios

Para esta parte creamos dos listas, `iterations` y `percentages` en las que guardamos el número de iteraciones y el porcentaje de puntos clasificados correctamente para cada vector aleatorio.

Ejecutamos el algoritmo para cada vector,
```python
for i in range(0, 10):
  w, its = ajusta_PLA(x, y, max_iters, np.random.rand(3))
  iterations.append(its)
  percentages.append(getPorc(x, y, clasifHiperplano(w)))
```
y finalmente mostramos la media en ambos casos,
```python
print("Vector inicial aleatorio (media de 10 ejecuciones)")
print('Iteraciones: {} épocas (± {:.02f})'
      .format(np.mean(iterations),np.std(iterations)))
print('% correctos: {}% (± {:.02f})'
      .format(np.mean(percentages),np.std(percentages)))
```

#### Valoración del resultado

En este apartado ejecutamos `testPLA(x_hom, y)`.
En todos los casos llegamos a clasificar de forma perfecta los puntos.

En el caso del vector cero, tarda un total de 3 épocas en converger, mientras que en el caso aleatorio tardamos de media $2.2 \pm 0.98$ épocas en converger.
Tarda menos en el caso aleatorio ya que el porcentaje de clasificación inicial será probablemente más alto.

Como vemos, el algoritmo es capaz de converger rápidamente hacia la solución óptima en este caso, ya que los datos son separables. Con otras semillas el algoritmo tarda más épocas pero es capaz de hacerlo correctamente.


### b) Hacer lo mismo que antes usando ahora los datos del apartado 2b de la sección.1. ¿Observa algún comportamiento diferente? En caso afirmativo diga cual y las razones para que ello ocurra.

En este apartado ejecutamos `testPLA(x_hom, y_noise)`.



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
  labels[i] = f(datos[i, 0], datos[i, 1], a, b)
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


#### Implementación

Para generar los datos de test seguimos el procedimiento análogo a la generación de los datos de training, es decir
```python
N_test = 1000
test = simula_unif(N_test, 2, intervalo)
test_hom = np.hstack((np.ones((N_test, 1)), test))

labels_test = np.empty((N_test, ))
for i in range(N_test):
  labels_test[i] = f(test[i, 0], test[i, 1], a, b)
```

A continuación debemos definir una función que calcule el error.
He decidido calcularlo de dos formas.
En primer lugar la log-verosimilitud 
(que es la función de la que obtenemos el gradiente utilizado en el algoritmo), dada por,
$$E(w) = \frac{1}{N} \sum_{n = 1}^N \log(1 + \exp(-y_nw^T\mathbf{x}_n)),$$
cuya traducción a código de NumPy es directa:
```python
def Err(w, x, y):
  return np.mean(np.log(1 + np.exp(-y*x.dot(w))))
```
Además, en segundo lugar también lo mostramos como porcentaje clasificado incorrectamente.

Por último mostramos el error, así como el porcentaje clasificado incorrectamente,
```python
print("% correctos en test RL: {}%".format(
  getPorc(test_hom, labels_test, clasifHiperplano(w))))
print("Error: {}".format(Err(w, test_hom, labels_test)))
```


#### Análisis de resultados

Redondeando a 5 cifras significativas, el error para la solución obtenida queda
$$E_{\operatorname{out}}(w) = 0.06741,$$
y el porcentaje de puntos de la muestra de test clasificados correctamente es de $99.3$%.

Como vemos el algoritmo proporciona muy buenos resultados en estas condiciones.
He incluido una representación de la recta con los puntos en la figura 10.

![Recta obtenida por la regresión logística](img/2.2.b.png){width=70%}

\newpage

# 3. Bonus

## Clasificación de dígitos

> (1.5 puntos) Clasificación de Dígitos. Considerar el conjunto de datos de los dígitos manuscritos 
> y seleccionar las muestras de los dígitos 4 y 8. Usar los ficheros de entrenamiento (training)
> y test que se proporcionan. Extraer las características de **intensidad promedio** y **simetría** en
> la manera que se indicó en el ejercicio 3 del trabajo 1.

### 1. Plantear un problema de clasificación binaria que considere el conjunto de entrenamiento como datos de entrada para aprender la función $g$.

Tal y como aparece en la plantilla, el objetivo es la clasificación en las clases de dígitos 4 y 8 (que representamos respectivamente por -1 y 1) a partir de sus características de intensidad promedio y simetría.

Este apartado ya está hecho en la plantilla proporcionada.
Los datos y etiquetas de training se guardan en `x` e `y` respectivamente, y los de test en `x_test` e `y_test`.

### 2. Usar un modelo de Regresión Lineal y aplicar PLA-Pocket como mejora. Responder a las siguientes cuestiones.

#### Implementación

Recordamos la función de error de la regresión lineal, que necesitábamos en la práctica 1,
```python
def Err(x, y, w):
  wN = np.linalg.norm(x.dot(w) - y)**2
  return wN/len(x)
```

Además, para regresión lineal podemos utilizar cualquiera de los dos algoritmos que utilizamos en la práctica anterior. Optamos por el algoritmo de la pseudoinversa.
```python
def pseudoinverse(x, y):
  u, s, v = np.linalg.svd(x)
  d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)
```

****

Podemos adaptar el algoritmo de `ajusta_PLA` para hacer la función `PLAPocket`, que tomará los mismos argumentos.

Añadimos dos variables, `w_best` y `err_best` que guardan el mejor clasificador encontrado hasta el momento y su valor de error, inicializados a
```python
w_best   = w
err_best = Err(datos, labels, w_best)
```

A continuación, en el bucle principal añadimos la comprobación de si el vector actual de pesos mejora la clasificación,
```python
err = Err(datos, labels, w)
if err < err_best:
  max_w = w
  err_best = err
```

El código completo de la función queda

```python
def PLAPocket(datos, labels, max_iter, vini):
  w = vini.copy()
  w_best   = w
  err_best = Err(datos, labels, w_best)

  for it in range(1, max_iter + 1):
    w_old = w.copy()

    for dato, label in zip(datos, labels):
      if signo(w.dot(dato)) != label:
        w += label*dato

    err = Err(datos, labels, w)
    if err < err_best:
      max_w = w
      err_best = err

    if np.all(w == w_old):  # No hay cambios
      return max_w, it

  return max_w, it
```

#### a) Generar gráficos separados (en color) de los datos de entrenamiento y test junto con la función estimada.

Hacemos tres estimaciones; la estimación de la pseudoinversa, la mejora de esta por `PLAPocket` y `PLAPocket` usando un vector inicial aleatorio.

Las ejecuciones de `PLAPocket` las hacemos con un máximo de 1000 iteraciones.
Los resultados pueden verse en las figuras 11 y 12.

![Ajuste lineal y PLA-Pocket sobre datos de dígitos manuscritos (training)](img/3.2.1.png){width=82%}

![Ajuste lineal y PLA-Pocket sobre datos de dígitos manuscritos (test)](img/3.2.2.png){width=82%}


Como vemos, la mejora de PLA-Pocket es indistinguible de la solución obtenida mediante regresión lineal.

#### b) Calcular $E_{\operatorname{in}}$ y $E_{\operatorname{test}}$ (error sobre los datos de test).

Los errores obtenidos pueden verse en la siguiente tabla:

| **Error** | Regresión Lineal | PLA-Pocket (RL) | PLA-Pocket (random) |
|-------|------------------|-----------------|---------------------|
| $E_{\operatorname{in}}$ | 0.90033 | 0.90033 | 6.29156 |
| $E_{\operatorname{test}}$ | 0.93745 | 0.93745 | 6.77846|

Como podíamos apreciar en la figura del apartado anterior, PLA-Pocket no consigue reducir el error de la regresión lineal ni consigue buenos resultados en el caso aleatorio.
Esto probablemente sea debido a que los datos del dataset con el que trabajamos no son separables.

#### c) Obtener cotas sobre el verdadero valor de $E_{\operatorname{out}}$ . Pueden calcularse dos cotas una basada en $E_{\operatorname{in}}$ y otra basada en $E_{\operatorname{test}}$ . Usar una tolerancia $\delta= 0.05$. ¿Que cota es mejor?

Calculo la cota sobre el error sólo en el caso PLA-Pocket (RL).

Aplicamos la fórmula de la cota del error, que podemos traducir a una función NumPy de forma directa.
La fórmula, para una clase de hipótesis finita $\mathcal{H}$ es
$$E_{\operatorname{out}}(h) \leq E_{\operatorname{in}}(h) + \sqrt{\frac{1}{2N} \log\left(\frac{2|\mathcal{H}|}{\delta}\right)}.$$

Si discretizamos el problema podemos ver que el número de hipótesis en la clase es el número de vectores de 3 flotantes de 64 bits, por lo que $|\mathcal{H}| = 3\cdot 2^{64}$ y la cota puede calcularse con la función,
```python
def bound(err, N, delta):
  return err + np.sqrt(1/(2*N)*(np.log(2/delta) + 3*64*np.log(2)))
```
Alternativamente podríamos usar la fórmula con la dimensión VC.

Para el cálculo de la cota sobre el error basta entonces calcular Ein y Etest y pasar como argumentos el tamaño de muestra (`N`) y la tolerancia (`delta`),
```python
Ein = Err(x, y, w_pla)
Etest = Err(x_test, y_test, w_pla)
Nin = len(x)
Ntest = len(x_test)
delta = 0.05

print("Apartado 3.2.c (en terminal)")
print("Cota superior de Eout (con Ein): {}".format(bound(Ein, Nin, delta)))
print("Cota superior de Eout (con Etest): {}".format(bound(
  Etest, Ntest, delta)))
```

Las cotas obtenidas son (redondeando a 5 cifras decimales)

- con $E_{\operatorname{in}}$, 0.88218 y
- con $E_{\operatorname{test}}$, 1.14097.

Es posible que las cotas fueran más informativas si usáramos la fórmula de la dimensión VC.
La cota a partir de los datos de test es más precisa, ya que es una muestra independiente de los datos utilizados para obtener el clasificador.

\newpage

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
xmin, xmax = np.min(x[:,0]), np.max(x[:,0])
ax.set_xlim(xmin, xmax)
ax.set_ylim(np.min(x[:,1]),np.max(x[:,1]))
```

A continuación, si no hay clases mostramos simplemente los puntos (`ax.scatter(x[:,0], x[:,1])`) y en otro caso los mostramos siguiendo el código que implementé para la práctica 0:

```python
class_colors = {-1 : 'green', 1 : 'blue'}
for cls, name in [(-1,"Clase -1"), (1,"Clase 1")]:
  # Obten los miembros de la clase
  class_members = x[y == cls]
  
  # Representa en scatter plot
  ax.scatter(class_members[:,0],
             class_members[:,1],
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



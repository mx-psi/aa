---
title: Práctica 1
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
documentclass: scrartcl
---

**Nota**: He establecido un formato unificado para los elementos de los vectores NumPy, con 3 cifras decimales.

# 1. Búsqueda iterativa de óptimos
## 1. Implementar el algoritmo de gradiente descendente

El algoritmo de gradiente descendente está implementado en la función `gradient_descent`.
Sus argumentos posicionales son:

- `initial_point`: Punto inicial
- `fun`: Función a minimizar. Debe ser diferenciable
- `grad_fun`: Gradiente de `fun`
- `eta`: Tasa de aprendizaje
- `max_iter`: Máximo número de iteraciones

Además tenemos un argumento opcional `errror2get`.
Si no pasamos este argumento hará `max_iter` iteraciones y si sí lo hacemos podrá parar antes de completar estas iteraciones si el valor de la función está por debajo de `error2get` (este criterio sólo será válido si la función tiene un mínimo en 0).

El código es el siguiente (omitiendo el comentario explicativo):

```python
def gradient_descent(initial_point, fun, grad_fun, eta,
                     max_iter, error2get = -math.inf):
  w = initial_point
  iterations = 0

  while fun(*w) > error2get and iterations < max_iter:
    w = w - eta*grad_fun(*w)
    iterations += 1

  return w, iterations
```

La variable `w` contiene la estimación actual del punto donde se alcanza el mínimo, que es inicializada con `initial_point`. `iterations` tiene el número de iteraciones.

A continuación aplicamos la fórmula del gradiente descendente (`w = w - eta*grad_fun(*w)`) hasta o bien quedarse por debajo del umbral `error2get` o bien llegar al número máximo de iteraciones.

En el caso de que no pasemos umbral este tomará el valor $-\infty$, siendo entonces la comprobación del umbral siempre cierta.

## 2. Encontrar el mínimo de $E$ usando gradiente descendente

Para analizar correctamente los resultados podemos demostrar analíticamente cuál es el valor mínimo que puede alcanzar la función y al menos uno de los puntos donde se alcanza.


### a) ¿Cuál es la expresión de $\nabla E$?

El gradiente de $E$ nos da las derivadas parciales de $E$ respecto a $u$ y $v$ en cada punto, esto es $\nabla E = (\frac{\partial E}{\partial u}, \frac{\partial E}{\partial v})$. Calculamos estas derivadas para obtener:

$$\nabla E(u,v) = (2 (u^2 e^v - 2 e^{-u} v^2) (2 e^{-u} v^2 + 2 u e^v), 2 (u^2 e^v - 4 e^{-u} v) (u^2 e^v - 2 e^{-u} v^2))$$

### b) ¿Cuántas iteraciones tarda el algoritmo en obtener un valor inferior a $10^{-14}$?

Como vemos en la ejecución, el algoritmo tarda un total de **33** iteraciones en alcanzar el mínimo.
El algoritmo para por tanto no por alcanzar el número máximo de iteraciones posibles sino porque el valor del punto está por debajo de la tolerancia máxima aceptable, $10^{-14}$.

### c) ¿En qué coordenadas se alcanza este valor?

Redondeando a 3 cifras decimales el mínimo obtenido se alcanza en el punto $(0.619, 0.968)$.

El valor de la función en este punto es, redondeando de nuevo a 3 cifras decimales, $5.997\cdot 10^{-15}$.

## 3. Encontrar el mínimo de $f$ usando gradiente descendente


### a) Minimizar $f$ con gradiente descendente variando $\eta$

Como en el ejercicio anterior, calculamos en primer lugar el gradiente, cuya expresión analítica es

$$\nabla f(x,y) = (2 x + 4 \pi \cos(2 \pi x) \sin(2 \pi y), 4 y + 4 \pi \sin(2 \pi x) \cos(2 \pi y))$$

TOOD discutir código

TODO plot de valor de la función

TODO discusión del plot

El mejor valor obtenido se da por tanto con tasa de aprendizaje $\eta = 0.01$, con la que, como podemos ver en el gráfico de contorno, estamos cerca de un mínimo global de la función. 

Del gráfico podemos ver que es mínimo local de la región representada, pero además sabemos que es mínimo global: cualquier mínimo global estará dentro de la elipse $x^2 + 2y^2 = 2$; fuera de esta $f(x,y) \geq x^2 + 2y^2 - 2 > 0$ que está por encima del mínimo. 

Además, la función verifica $f(x,y) = f(-x,-y)$, luego habrá al menos dos mínimos globales, y nuestro método se aproxima a uno de ellos.

![Gráfico de contorno de la función $f$ con el mínimo obtenido marcado en rojo.](img/1.3.a.png)

### b) Obtener el valor mínimo en función del punto de inicio

Para este apartado utilizamos la función `gradient_descent` definida anteriormente para calcular el gradiente descendiente. Creamos un iterable de puntos iniciales `initial_points` de acuerdo a los requeridos en el enunciado.

A continuación imprimimos en formato tabla los resultados, ejecutando la función `gradient_descent` con cada punto inicial. La tasa de aprendizaje elegida es 0.01 y el número de iteraciones 50, de la misma forma que el apartado anterior.

```python
initial_points = map(np.array, [[0.1,0.1], [1,1], [-.5, -.5], [-1, -1]])

print("{:^15}  {:^15}  {:^7}".format("Inicial","Final","Valor"))
for initial in initial_points:
  w, _ = gradient_descent(initial, f, gradf, eta, maxIter)
  print("{}  {}  {: 1.3f}".format(initial, w, f(*w)))
```

La primera llamada a `print` imprime los nombres de las columnas alineados.
Los resultados pueden verse en la siguiente tabla:

| $\mathbf{(x_0,y_0)}$ | $\mathbf{(x_{\operatorname{min}},y_{\operatorname{min}})}$ | $\mathbf{E(x_{\operatorname{min}},y_{\operatorname{min}})}$ | 
|----------------------|------------------------------------------------------------|-------------------------------------------------------------|
| $(0.1, 0.1)$         | $( 0.244, -0.238)$                                         |   -1.820                                                    |
| $(1, 1)$             | $( 1.218,  0.713)$                                         |    0.593                                                    |
| $(-0.5, -0.5)$       | $(-0.731, -0.238)$                                         |   -1.332                                                    |
| $(-1, -1)$           | $(-1.218, -0.713)$                                         |    0.593                                                    |

Como vemos, incluso con la tasa de aprendizaje que da mejores resultados, esto es, 0.01, el resultado depende enormemente del punto inicial.

En primer lugar podemos observar la simetría de los resultados con los puntos iniciales $(1,1)$ y $(-1,-1)$.
Esto se debe a que la función verifica $f(x,y) = f(-x,-y)$ para cualesquiera $x,y\in \mathbb{R}$.

El mejor resultado se obtiene en el punto $(0.1,0.1)$ que, como ya comentamos en el apartado anterior,
encuentra un punto cercano a uno de los óptimos.

En el resto de casos nos quedamos en puntos muy alejados a uno de los óptimos.
Esto se debe, como podemos apreciar en el gráfico de contorno del apartado anterior, a que es una función con gran cantidad de máximos locales y por tanto el método de gradiente descendente no será capaz de escapar de estos, al menos si la tasa de aprendizaje es constante.

## 4. ¿Cuál es la conclusión para encontrar el mínimo de una función arbitraria?


# 2. Regresión lineal
## 1. Estimación de un modelo de regresión lineal
### Implementación usando pseudoinversa
### Implementación usando SGD
### Pintar las soluciones obtenidas
### Valorar la bondad del resultado
## 2. Realizar el siguiente experimento
### a) Generar y mostrar muestra de entrenamiento
### b) Generar mapa de etiquetas usando $f$
### c) Ajustar un modelo de regresión lineal usando SGD
### d) Ejecutar el experimento 1000 veces
### e) Valoración

# Bonus

## 1. Repetir experimento haciendo uso del método de Newton
### Implementación del algoritmo
### Generar un gráfico de cómo desciende el valor de la función
### Extraer conclusiones y comparar

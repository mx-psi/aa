---
title: Cuestionario 1
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
toc-depth: 1
toc-title: Índice
header-includes:
- \usepackage{stmaryrd}
- \newcommand{\cov}{\operatorname{cov}}
- \newcommand{\x}{\mathbf{x}}
- \newcommand{\w}{\mathbf{w}}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
---

\newpage

# Pregunta 1

> Identificar, para cada una de las siguientes tareas, 
>
> 1. cuál es el problema, 
> 2. qué tipo de aprendizaje es el adecuado (supervisado, no supervisado, refuerzo) y
> 3. los elementos de aprendizaje ($\mathcal{X}, f, \mathcal{Y}$) 
>
> que deberíamos usar en cada caso. Si una tarea se ajusta a más de un tipo, explicar cómo
> y describir los elementos para cada tipo.

### a) Clasificación automática de cartas por distrito postal.

El problema está en la obtención del código postal de la carta, por ejemplo a partir de una foto de la misma.
Para ello podemos enfocarlo como un problema de clasificación de los distintos dígitos del código postal.

El tipo de aprendizaje en este caso es supervisado: a partir de ejemplos manuscritos de los dígitos del código postal y el número que representan debemos obtener una función que los clasifica.

Los elementos del aprendizaje son:

- $\mathcal{X}$, el espacio de características, es el conjunto de imágenes de dígitos manuscritos,
- $\mathcal{Y}$, el conjunto de etiquetas, es el conjunto de los dígitos $\{0,1,2,3,4,5,6,7,8,9\}$ y
- $f : \mathcal{X} \to \mathcal{Y}$ será la función que lleva cada imagen al dígito que representa.

###  b) Decidir si un determinado índice del mercado de valores subirá o bajará dentro de un periodo de tiempo fijado.

El problema se basa en predecir, basándose en los datos del mercado de valores, cuál será la evolución de un determinado índice después de un periodo de tiempo fijo.

Podemos enfocarlo desde el paradigma del aprendizaje supervisado. El espacio de características $\mathcal{X}$ sería el espacio de datos económicos que tenemos disponibles para la predicción. El problema puede tratarse dentro de este enfoque como un problema de clasificación, en cuyo caso las etiquetas serían la subida (+1) o bajada (-1), es decir, $\mathcal{Y} = \{-1,+1\}$ o como un problema de regresión, prediciendo, por ejemplo, el porcentaje de subida o bajada, para el que tomaríamos  $\mathcal{Y} = \mathbb{R}$.

La función a aproximar sería $f: \mathcal{X} \to \mathcal{Y}$, que nos da dados unos datos económicos, una predicción de si el índice sube o baja (o de cuánto lo hace en el caso de la regresión).

### c) Hacer que un dron sea capaz de rodear un obstáculo.

Este problema consiste en diseñar un movimiento con el cuál un dron sea capaz de moverse esquivando un obstáculo.

Para ello creo que el enfoque más adecuado es el de aprendizaje por refuerzo, ya que, en este caso, si bien sí podemos indicar si el dron ha realizado la acción de forma correcta, la forma más adecuada de hacerlo es por medio de un sistema de recompensa en función de si las acciones del dron han conducido al objetivo deseado. 

En este caso, al no tratarse de aprendizaje supervisado, no tiene sentido hablar de los elementos del aprendizaje.

### d) Dada una colección de fotos de perros, posiblemente de distintas razas, establecer cuántas razas distintas hay representadas e la colección

El problema consiste en la obtención de información sobre la estructura geométrica del espacio de fotos de perros (*clustering*).

En este caso estamos claramente ante un problema de aprendizaje no supervisado en el que el objetivo no es la clasificación o regresión, sino la separación en clases de los datos. 
No podemos adecuarlo por tanto a los elementos de aprendizaje del paradigma supervisado, ya que si bien sí tenemos un dominio del problema $\mathcal{X}$ (el conjunto de las fotos de perros), no conocemos las posibles etiquetas $\mathcal{Y}$.

# Pregunta 2

>  ¿Cuales de los siguientes problemas son más adecuados para una aproximación por aprendizaje y cuales más 
> adecuados para una aproximación por diseño? Justificar la decisión.

### a) Determinar si un vertebrado es mamífero, reptil, ave, anfibio o pez.

Estamos ante un problema en la que la aproximación más adecuada es el **aprendizaje** supervisado, ya que no existe un procedimiento algorítmico claro que permita determinar la clase a la que pertenezca un vertebrado a partir de una foto u otros datos similares. 

A partir de un conjunto de fotos correctamente clasificadas de estos vertebrados podremos aprender una función de clasificación que nos indique la clase de cada uno.

### b) Determinar si se debe aplicar una campaña de vacunación contra una enfermedad.

En este caso, dado que las campañas de vacunación tienen un procedimiento claro y estandarizado, creo que el enfoque más adecuado es el de **diseño**: estableceremos la campaña de vacunación transformando los protocolos ya existentes en un procedimiento algorítmico.

### c) Determinar perfiles de consumidor en una cadena de supermercados.

Se trata de un problema de **aprendizaje** (en un primer momento no supervisado) en el que tenemos que dividir a los consumidores (definidos por un vector de características en función de sus hábitos de consumo) en distintas clases.

No podemos enfocarlo mediante el diseño ya que no conocemos qué clases hay ni cómo obtenerlas; tampoco se trata de aprendizaje supervisado porque no contamos con ejemplos ya clasificados.

### d) Determinar el estado anímico de una persona a partir de una foto de su cara.

Es claramente un problema de **aprendizaje** supervisado, en el que podemos entrenar un modelo un conjunto de ejemplos de fotos y su estado anímico; este enfoque puede hacerse por categorías y por tanto como un problema de clasificación o de forma continua estableciendo una escala numérica de estado anímico.

### e) Determinar el ciclo óptimo para las luces de los semáforos en un cruce con mucho tráfico.

Este caso puede enfocarse como un problema de **diseño**, ya que tenemos un procedimiento algorítmico claro que resuelve el problema. Alternativamente podemos enfocarlo como un problema de aprendizaje por refuerzo en el que las acciones a tomar son el cambio de las luces del semáforo y la recompensa se da en función del tiempo medio de espera de los vehículos.

\newpage

# Pregunta 3

> Construir un problema de aprendizaje desde datos para un problema de clasificación de
> fruta en una explotación agraria que produce mangos, papayas y guayabas. Identificar los
> siguientes elementos formales $\mathcal{X} , \mathcal{Y}, \mathcal{D}, f$ del problema. 
> Dar una descripción de los mismos que pueda ser usada por un computador. 
> ¿Considera que en este problema estamos ante un caso de etiquetas con ruido o sin ruido? 
> Justificar las respuestas.

Podemos construir un problema de aprendizaje supervisado en esta situación recolectando características de las frutas como fotografías, su rugosidad, su dureza o su tamaño. Los elementos formales serían

- $\mathcal{X}$ sería el espacio de características, incluyendo por ejemplo las anteriormente mencionadas.
  Para permitir el tratamiento informático deberíamos transformar estas características en variables continuas (como 
  es el caso de la dureza, tamaño o las imágenes representadas como matrices) o discretas (como en el caso de la 
  rugosidad).
- $\mathcal{Y}$ sería el conjunto de las etiquetas, $\mathcal{Y} = \{\operatorname{mangos}, \operatorname{papayas},\operatorname{guayabas}\}$. Para su representación podemos escoger un conjunto numérico $\{-1,0,1\}$ o de vectores $\{(1,0,0), (0,1,0), (0,0,1)\}$.
- $\mathcal{D}$ sería el conjunto de datos de ejemplo, esto es un subconjunto de $\mathcal{X} \times \mathcal{Y}$ 
  que incluya ejemplos de las distintas frutas y sus características, i.e., una fotografía y diferentes medidas.
  
Considero que estamos ante un caso de etiquetas con ruido, dado que puede haber errores en la creación de $\mathcal{D}$ en la que las frutas estén incorrectamente clasificadas.

# Pregunta 4

>  Suponga una matriz cuadrada $A$ que admita la descomposición $A = X^TX$ para alguna
> matriz $X$ de números reales. Establezca una relación entre los valores singulares de las
> matriz $A$ y los valores singulares de $X$.

Sea $X \in \mathcal{M}_{n\times m}(\mathbb{R})$. 
$X$ admite descomposición en valores singulares de la forma $$X = U D V^T$$ con 
$U,V$ matrices ortogonales y
$D$ matriz diagonal.
Los elementos de la diagonal de $D$ serán los valores singulares de $X$.

Utilizando esta descomposición tenemos que $$A = XX^T = (U DV^T) (U DV^T)^T = U DV^TVD^TU^T = UDD^TU^T,$$
donde hemos utilizado que $(A_1A_2)^T = A_2^TA_1^T$ y que la inversa de una matriz ortogonal es su traspuesta.

Esta es una descomposición en valores singulares, ya que como $U$ es ortogonal, también lo será $U^T$.
Además, $DD^T$ es diagonal por ser $D$ y $D^T$ diagonales: $$(DD^T)_{ij} = \sum_{k = 1}^m D_{ik}D_{jk} = \begin{cases}0 & \text{ si } i \neq j \text{ y } \\ D_{ii}^2 & \text{ si } i = j\end{cases}.$$

Por tanto tenemos que los valores singulares de $A$ serán los valores de la diagonal de $DD^T$, es decir:
los valores singulares de $A$ serán **los cuadrados de los valores singulares de $X$**.

# Pregunta 5

> Sean  $\x$ e $\mathbf{y}$ dos vectores de características de dimensión $M \times 1$. La expresión
> $$\cov(\x,\mathbf{y}) = \frac{1}{M}\sum_{i = 1}^M (x_i - \bar{x})(y_i - \bar{y})$$
> define la covarianza entre dichos vectores, donde $\bar{z}$ representa el valor medio de los elementos
> de $\mathbf{z}$. Considere ahora una matriz $X$ cuyas columnas representan vectores de características.
> La matriz de covarianzas asociada a la matriz $X = (\x_1 , \x_2 , \dots , \x_N)$ es el 
> conjunto de covarianzas definidas por cada dos de sus vectores columnas. Es decir,
> $$\cov(X) = \left(\begin{matrix}
  \cov(\x_1,\x_1) & \cov(\x_1, \x_2) & \cdots & \cov(\x_1,\x_N) \\
  \cov(\x_2,\x_1) & \cov(\x_2, \x_2) & \cdots & \cov(\x_2,\x_N) \\
  \vdots          &  \vdots          & \ddots &  \vdots \\
 \cov(\x_N,\x_1) & \cov(\x_N, \x_2) & \cdots & \cov(\x_N,\x_N)
\end{matrix}\right)$$
> Sea $\mathbf{1}^T_M = (1,1,\dots,1)$ un vector $M \times 1$ de unos. 
> Mostrar qué representan las siguientes expresiones.

### a) $E_1 = \mathbf{1}\mathbf{1}^TX$

Notamos por $A$ a la matriz cuadrada de orden $M$ dada por $A = \mathbf{1}\mathbf{1}^T$, que verifica $A_{ij} = 1$.

$E_1$ será una matriz $M \times N$, en particular, en la posición $i \in \{1,\dots,M\}, j \in \{1,\dots, N\}$ tenemos que $$(E_1)_{ij} = \sum_{k=1}^M A_{ik}X_{kj} = \sum_{k=1}^M X_{kj} = \sum_{k=1}^M (\x_j)_k,$$
es decir, si $\operatorname{suma}(\x_j) = \sum_{k=1}^M (\x_j)_k$, $E_1$ puede expresarse como 
$$E_1 = \left(\begin{matrix}
  \operatorname{suma}( \x_1)  & \operatorname{suma}( \x_2) & \cdots & \operatorname{suma}(\x_N) \\
  \operatorname{suma}(\x_1) & \operatorname{suma}( \x_2) & \cdots & \operatorname{suma}(\x_N) \\
  \vdots          &  \vdots          & \ddots &  \vdots \\
 \operatorname{suma}(\x_1) & \operatorname{suma}( \x_2) & \cdots & \operatorname{suma}(\x_N)
\end{matrix}\right).$$

### b) $E_2 = (X-\frac{1}{M}E_1)^T(X - \frac{1}{M}E_1)$

Notamos $C := X-\frac{1}{M}E_1$.

En primer lugar vemos que, como cada columna tiene $M$ componentes,
$$\frac{1}{M}E_1 = \left(\begin{matrix}
  \bar{ \x_1}  & \bar{ \x_2} & \cdots & \bar{\x}_N \\
  \bar{\x}_1 & \bar{ \x_2} & \cdots & \bar{\x}_N \\
  \vdots          &  \vdots          & \ddots &  \vdots \\
 \bar{\x}_1 & \bar{ \x_2} & \cdots & \bar{\x}_N
\end{matrix}\right),$$
por lo que, expresada por columnas, $$C = X-\frac{1}{M}E_1 = (\x_1 - \bar{\x}_1\mathbf{1}, \dots, \x_N - \bar{\x}_N\mathbf{1}).$$

Por tanto, $$(E_2)_{ij} = (C^TC)_{ij} = \sum_{k=1}^M C^T_{ik}C_{kj} = \sum_{k=1}^M C_{ki}C_{kj} = \sum_{k=1}^M ((\x_i)_k - \bar{\x}_i)((\x_j)_k - \bar{\x}_j) = M \cov(\x_i, \x_j),$$
luego tenemos que $E_2 = M \cov(X)$, es decir, $E_2$ representa **$M$ veces la matriz de covarianza**.

# Pregunta 6

> Considerar la matriz **hat** definida en regresión, $\hat{H} = X(X^TX)^{-1} X^T$ , donde $X$ es la matriz
> de observaciones de dimensión $N \times (d + 1)$, y $X^TX$ es invertible.

### a) ¿Que representa la matriz $\hat{H}$ en un modelo de regresión?

La matriz $\hat{H}$ es la matriz de proyección del modelo de regresión, y nos permite calcular los valores estimados para los datos de entrada.

En concreto, si $y$ es el vector de salida para los datos $X$, sabemos que $$\hat{y} = \hat{H}y = X(X^TX)^{-1}X^Ty = Xw,$$ donde $w$ es el vector de pesos calculado en el modelo de regresión. Por tanto $\hat{H}y$ nos da los valores estimados para cada dato de $X$.

### b) Identifique la propiedad más relevante de dicha matriz en relación con regresión lineal.

Su propiedad más relevante es la idempotencia, esto es, $\hat{H}^2 = \hat{H}$, ya que esto nos da una noción de consistencia: si aplicamos el modelo de regresión con los valores $\hat{y}$ estimados el modelo nos devolverá exactamente las mismas estimaciones.

# Pregunta 7

> La regla de adaptación de los pesos del Perceptron 
> ($\w_{\operatorname{new}} = \w_{\operatorname{old}} + y\x$) 
> tiene la interesante propiedad de que mueve el vector de pesos en la dirección adecuada 
> para clasificar $\x$ de forma correcta. 
> Suponga el vector de pesos $\w$ de un modelo y un dato $\x(t)$ mal clasificado
> respecto de dicho modelo. Probar matemáticamente que el movimiento de la regla de
> adaptación de pesos siempre produce un movimiento de $\w$ en la dirección correcta para
> clasificar bien $\x(t)$.

Sea $\w$ un vector asociado a un modelo lineal, esto es, un vector que clasifica cada $\x$ en las clases $\{-1,1\}$ mediante la función $\operatorname{signo}(\w^T\x)$, que nos indica a qué lado del hiperplano con vector normal $\w$ que pasa por $0$ queda el vector $\x$.


Sea $\x$ un dato con etiqueta $y$ mal clasificado, esto es, tal que $\operatorname{signo}(\w^T\x) \neq y$.
En este caso, en el algoritmo de Perceptron, aplicamos $\w_{\operatorname{new}} = \w_{\operatorname{old}} + y\x$.
El nuevo producto será, por linealidad, $$\w_{\operatorname{new}}^T\x = (\w_{\operatorname{old}} + y\x)^T\x = \w_{\operatorname{old}}^T \x + y \x^T\x.$$

Sabemos que $\x^T\x = \lVert \x\rVert^2 > 0$ (no será nulo porque usamos coordenadas homogéneas).
Supongamos que $y = 1$. En tal caso $y \x^T\x > 0$, y por tanto el cambio del vector se hace en la dirección correcta (nos acercamos a que $\w^T\x > 0$).
Análogamente, si suponemos $y = -1$ tendremos $y \x^T\x < 0$ y por tanto nos acercaremos a $\w^T\x < 0$.

Es decir, en ambos casos, si un vector $\x$ está mal clasificado, la actualización del vector de pesos lo mueve en la dirección adecuada para acercarse al signo correcto.

\newpage

# Pregunta 8

>  Sea un problema probabilístico de clasificación binaria con etiquetas $\{0,1\}$, es decir
> $P(Y = 1) = h(\x)$ y $P(Y = 0) = 1 - h(\x)$, para una función $h$ dependiente de la muestra.

### Apartado a)

> Considere una muestra i.i.d. de tamaño $N$ $(\x_1 , \dots, \x_N)$. Mostrar que la función $h$ que maximiza la verosimilitud de la muestra es la misma que minimiza
> $$E_{\operatorname{in}}(\w) = \sum_{n=1}^N \llbracket y_n = 1\rrbracket \operatorname{ln}\frac{1}{h(\x_n)} + \llbracket y_n = 0\rrbracket \operatorname{ln}\frac{1}{1-h(\x_n)}$$
> donde $\llbracket\cdot\rrbracket$ vale 1 o 0 según que sea verdad o falso respectivamente la expresión en su interior.

Hago explícita en la notación de este ejercicio la dependencia de $\w$ de la distribución de probabilidad considerada.

La verosimilitud de la muestra dada es $$L(\w) = \prod_{n = 1}^N P(y_n | \x_n),$$ donde hemos utilizado que la muestra es independiente (para justificar el producto) e idénticamente distribuida (con distribución dada por $P(\cdot)$).

Utilizando la notación del enunciado, podemos expresar este producto de la siguiente forma $$L(\w) = \prod_{n = 1}^N h(\x_n)^{\llbracket y_n = 1\rrbracket}(1 - h(\x_n))^{\llbracket y_n = 0\rrbracket},$$
ya que $P(y_n = 1) = h(\x_n)$ y $P(y_n = 0) = 1 - h(\x_n)$.

Maximizar esta verosimilitud será equivalente a minimizar $$- \frac{1}{n}\operatorname{ln}\left(L(\w)\right),$$
ya que la función $f(x) = - \frac{1}{n}\operatorname{ln}\left(x\right)$ es decreciente.
Vemos por último que $E_{\operatorname{in}}(\w) = -\frac{1}{n}\operatorname{ln}\left(L(\w)\right).$

En efecto, tenemos que 
$$- \frac{1}{n}\operatorname{ln}\left(L(\w)\right) = - \frac{1}{n}\operatorname{ln}\left( \prod_{n = 1}^N h(\x_n)^{\llbracket y_n = 1\rrbracket}(1 - h(\x_n))^{\llbracket y_n = 0\rrbracket}\right).$$
Como el logaritmo transforma productos en sumas
$$- \frac{1}{n}\operatorname{ln}\left(L(\w)\right) = - \frac{1}{n}\sum_{n = 1}^N \operatorname{ln}\left(h(\x_n)^{\llbracket y_n = 1\rrbracket}\right) + \operatorname{ln}\left(1 - h(\x_n))^{\llbracket y_n = 0\rrbracket}\right).$$

A continuación, utilizando que $\operatorname{ln}(x^n) = n\operatorname{ln}(x)$,
$$- \frac{1}{n}\operatorname{ln}\left(L(\w)\right) = - \frac{1}{n}\sum_{n = 1}^N {\llbracket y_n = 1\rrbracket}\operatorname{ln}\left(h(\x_n)\right) + {\llbracket y_n = 0\rrbracket}\operatorname{ln}\left(1 - h(\x_n)\right),$$
y por último metemos el signo negativo de la expresión dentro de los logaritmos, obteniendo el error
$$- \frac{1}{n}\operatorname{ln}\left(L(\w)\right) = \frac{1}{n}\sum_{n = 1}^N {\llbracket y_n = 1\rrbracket}\operatorname{ln}\left(\frac{1}{h(\x_n)}\right) + {\llbracket y_n = 0\rrbracket}\operatorname{ln}\left(\frac{1}{1 - h(\x_n)}\right) = E_{\operatorname{in}}(\w).$$

Por tanto minimizar $E_{\operatorname{in}}(\w)$ es equivalente a maximizar $L(\w)$.

### Apartado b)

>  Para el caso $h(\x) = \sigma(\w^T \x)$ mostrar que minimizar el error de la muestra en el
>  apartado anterior es equivalente a minimizar el error muestral
> $$E_{\operatorname{in}}(\w) = \frac{1}{N} \sum_{n=1}^N \operatorname{ln}\left(1 + \exp(-y_n\w^T\x_n)\right)$$

Asumo que para este apartado las etiquetas son $\{-1,1\}$ y no $\{0,1\}$.

Para cualquier $x\in \mathbb{R}$ podemos comprobar que
$$1 - \sigma(x) = \frac{1+ e^{-x}}{1 + e^{-x}} - \frac{1}{1 + e^{-x}} = \frac{e^{-x}}{1 + e^{-x}} = \sigma(-x),$$
y por tanto
$$1 - h(\x) = 1 - \sigma(\w^T\x) = \sigma(-w^T\x) = h(-\x),$$
de lo que deducimos $P(Y = y | \x) = h(y\x),$ por lo que la expresión del apartado anterior puede simplificarse a
$$E_{\operatorname{in}}(\w) = \sum_{n=1}^N \operatorname{ln}\frac{1}{h(y_n\x_n)}$$

Por último tenemos que
$$\operatorname{ln}\left(\frac{1}{h(y_n\x_n)}\right) = \operatorname{ln}\left(\frac{1}{\sigma(y_n\w^T \mathbf{x_n})}\right)  =  \operatorname{ln}\left(\left(\frac{1}{1 + \exp(-y_n\w^T \mathbf{x_n})}\right)^{-1}\right) = \operatorname{ln}\left(1 + \exp(-y_n\w^T \mathbf{x_n})\right),$$
con lo que uniendo las dos igualdades tenemos el resultado buscado
$$E_{\operatorname{in}}(\w) = \frac{1}{N} \sum_{n=1}^N \operatorname{ln}\left(1 + \exp(-y_n\w^T\x_n)\right).$$


# Pregunta 9

>  Derivar el error Ein para mostrar que en regresión logística se verifica:
> $$\nabla E_{\operatorname{in}}(\w) = - \frac{1}{N}\sum_{n=1}^N \frac{y_n \x_n}{1 + \exp(y_n\w^T\x_n)} = \frac{1}{N}\sum_{n=1}^N -y_n \x_n\sigma(-y_n\w^T\x_n)$$
> Argumentar sobre si un ejemplo mal clasificado contribuye al gradiente más que un ejemplo
> bien clasificado.

### Gradiente del error

Como en el ejercicio anterior, noto $\exp(x) = e^x$.

Calculamos la derivada con respecto de $\w_i$.
En primer lugar, como la derivada es un operador lineal tenemos que
$$\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i} = \frac{1}{N} \sum_{n=1}^N \frac{\partial}{\partial w_i}\left(\operatorname{log}(1 + \exp(-y_n\w^T\x_n))\right)$$

Para derivar esta expresión interior utilizamos la regla de la cadena y utilizando las derivadas del logaritmo y la exponencial tenemos que 
\begin{align*}
\frac{\partial}{\partial w_i}\left(\operatorname{log}(1 + \exp(-y_n\w^T\x_n))\right) & = \frac{\exp(-y_n\w^T\x_n)}{1 + \exp(-y_n\w^T\x_n)} \cdot \frac{\partial}{\partial w_i}\left(-y_n\w^T\x_n\right) \\ 
& = \sigma(-y_n\w^T\x_n) \cdot \frac{\partial}{\partial w_i}\left(-y_n\w^T\x_n\right)
\end{align*}

Para esta última expresión vemos que $$\w^T\x_n = \sum_{i=1}^M \w_i(\x_n)_i,$$
luego tenemos que $$\frac{\partial}{\partial w_i}\left(-y_n\w^T\x_n\right) = -y_n(\x_n)_i.$$

Juntando estas ecuaciones obtenemos 
$$\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i} = \frac{1}{N} \sum_{n=1}^N -y_n\sigma(-y_n\w^T\x_n)(\x_n)_i,$$
o expresándolo en términos del gradiente
$$\nabla E_{\operatorname{in}}(w) = \left(\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i}\right)_{i=1\dots M} = \frac{1}{N} \sum_{n=1}^N  -y_n\sigma(-y_n\w^T\x_n)\x_n.$$

### Contribución al gradiente de ejemplos bien y mal clasificados

De nuevo asumo que las etiquetas son $\{-1,1\}$.

En tal caso, en un ejemplo bien clasificado se verifica $-y_n\w^T\x_n < 0$ mientras que en un ejemplo mal clasificado tenemos que $-y_n\w^T\x_n > 0$.

La función logística $\sigma$ cumple, para $x > 0$, $$\sigma(x) = \frac{1}{1+ e^{-x}} > \frac{1}{1 + e^x} = \sigma(-x),$$ ya que $e^{-x} < e^x$ para $x > 0$.

Juntando estas dos informaciones deducimos que **los ejemplos mal clasificados contribuyen más al error**, ya que tienen $\sigma$ tomará valores mayores para los ejemplos mal clasificados (donde se evalúa un punto positivo) que en los bien clasificados (donde se evalúa en un punto negativo).

# Pregunta 10

> Definamos el error en un punto $(\x_n, y_n )$ por
> $$\mathbf{e}_n (\w) = \max(0, -y_n \w^T \x_n)$$
> Argumentar si con esta función de error el algoritmo PLA puede interpretarse como SGD
> sobre $\mathbf{e}_n$ con tasa de aprendizaje $\nu = 1$.

**Sí**, el algoritmo PLA puede interpretarse de esta forma.

Podemos definir $\mathbf{e}_n$ por casos como $$\mathbf{e}_n(\w) = \begin{cases} -y_n \w^T \x_n & \text{ si } \operatorname{signo}(\w^T \x_n) \neq y_n \text{ o} \\ 0 & \text{ si } \operatorname{signo}(\w^T \x_n) = y_n\end{cases},$$
ya que $\operatorname{signo}(\w^T \x_n) \neq y_n \iff y_n\w^T \x_n < 0$.

$\mathbf{e}_n$ no es necesariamente derivable en el hiperplano $\{\w \;:\; y_n\w^T\x_n = 0\}$, pero podemos tomar la derivada en cualquier dirección y el algoritmo SGD seguirá funcionando. Si $\operatorname{signo}(\w^T \x_n) \neq y_n$, entonces $$\frac{\partial \mathbf{e}_n(\w)}{\partial \w_i} = -y_n(\x_n)_i, \quad \forall i \in \{1, \dots, m\}$$ 
luego el gradiente de $\mathbf{e}_n$ es por tanto $$\nabla \mathbf{e}_n(\w) = \begin{cases} -y_n \x_n & \text{ si } \operatorname{signo}(\w^T \x_n) \neq y_n \text{ o} \\ 0 & \text{ si } \operatorname{signo}(\w^T \x_n) = y_n\end{cases}.$$

La ecuación que da el proceso iterativo del gradiente descendente es 
$$\w_{n+1} := \w_n - \nu \nabla \mathbf{e}_n(\w_n).$$
Sustituyendo el gradiente calculado anteriormente,
$$\w_{n+1} = \begin{cases} \w_n -y_n \x_n & \text{ si } \operatorname{signo}(\w^T \x_n) \neq y_n \text{ o} \\ \w_n & \text{ si } \operatorname{signo}(\w^T \x_n) = y_n\end{cases},$$
que es la expresión utilizada en el algoritmo PLA.

\newpage

# BONUS 1

>  En regresión lineal con ruido en las etiquetas, el *error fuera de la muestra para una $h$ dada* puede expresarse como
> $$E_{\operatorname{out}}(h) = \mathbb{E}_{\x,y}[(h(\x) - y)^2] = \iint (h(\x) - y)^2p(\x, y)\mathrm{d}\x\mathrm{d}y.$$

### Apartado a)

> Desarrollar la expresión y mostrar que
> $$E_{\operatorname{out}}(h) = \int \left( h(\x)^2\int p(y|\x)\mathrm{d}y - 2h(\x)\int y \cdot p(y | \x)\mathrm{d}y + \int y^2p(y|\x)\mathrm{d}y \right)p(\x)\mathrm{d}\x.$$

En primer lugar vemos que
$$\iint (h(\x) - y)^2p(\x, y)\mathrm{d}\x\mathrm{d}y = \iint (h(\x)^2 - 2h(\x)y + y^2)p(y|\x)p(\x)\mathrm{d}\x\mathrm{d}y,$$
sin más que desarrollar el producto y utilizar la fórmula para obtener la distribución condicionada a partir de la distribución conjunta $p(\x,y)$.

A continuación distribuimos el producto
$$\iint (h(\x)^2p(y|\x) - 2h(\x)yp(y|\x) + y^2p(y|\x))p(\x)\mathrm{d}\x\mathrm{d}y,$$
y reordenamos las integrales haciendo uso del teorema de Fubini y de la linealidad de la integral, obteniendo
$$\int \left(\int h(\x)^2p(y|\x)\mathrm{d}y - \int 2h(\x)y\cdot p(y|\x)\mathrm{d}y + \int y^2p(y|\x)\mathrm{d}y \right)p(\x)\mathrm{d}\x.$$

Por último basta ver que en las dos primeras integrales podemos sacar los términos que no dependen de $y$ como constantes, llegando a la expresión deseada, $$E_{\operatorname{out}}(h) = \int \left( h(\x)^2\int p(y|\x)\mathrm{d}y - 2h(\x)\int y \cdot p(y | \x)\mathrm{d}y + \int y^2p(y|\x)\mathrm{d}y \right)p(\x)\mathrm{d}\x.$$


### Apartado b)

> El término entre paréntesis en Eout corresponde al desarrollo de la expresión
> $$\int (h(\x)-y)^2p(y|\x)\mathrm{d}y$$
> ¿Que mide este término para una $h$ dada?.

Este término mide, para un $\x$ fijo, el valor medio (esperanza) del error al cuadrado $(h(\x)  - y)^2$ respecto de la distribución condicionada de $y$ fijado $\x$ (cuya función de densidad es $p(y|\x)$).

Es decir, para un cierto dato $\x$ tendremos cuál será el error medio del valor que predice $h$.

### Apartado c)

> El objetivo que se persigue en Regression Lineal es encontrar la función $h \in \mathcal{H}$ que
> minimiza $E_{\operatorname{out}}(h)$. Verificar que si la distribución de probabilidad $p(\x, y)$ con la que
> extraemos las muestras es conocida, entonces la hipótesis óptima $h^\ast$ que minimiza
> $E_{\operatorname{out}}(h)$ está dada por
> $$h^\ast(\x) = \mathbb{E}_y[y|\x] = \int y \cdot p(y|\x)\mathrm{d}y.$$

En efecto, si partimos de la expresión obtenida en el apartado a) y simplificamos tenemos que
\begin{align*}
E_{\operatorname{out}}(h) & = \int \left( h(\x)^2\int p(y|\x)\mathrm{d}y - 2h(\x)\int y \cdot p(y | \x)\mathrm{d}y + \int y^2p(y|\x)\mathrm{d}y \right)p(\x)\mathrm{d}\x \\
& = \int \left( h(\x)^2 - 2h(\x)h^\ast(\x) + \mathbb{E}_y[y^2|\x]\right)p(\x)\mathrm{d}\x
\end{align*}
donde hemos usado que la integral de la función de densidad condicionada vale 1 y la definición de esperanza y de $h^\ast$.

A continuación completamos el cuadrado sumando y restando $h^\ast(\x)^2$
$$E_{\operatorname{out}}(h) = \int \left( h(\x)^2 - 2h(\x)h^\ast(\x) + h^\ast(\x)^2 + \mathbb{E}_y[y^2|\x] - h^\ast(\x)^2\right)p(\x)\mathrm{d}\x,$$
y tras agrupar el cuadrado y sustituir en el lado derecho por la definición de $h^\ast$:
\begin{align*}
E_{\operatorname{out}}(h) & = \int (h(\x) - h^\ast(\x))^2p(\x)\mathrm{d}\x & + & \int \left(\mathbb{E}_y[y^2|\x] - \mathbb{E}_y[y|\x]^2\right)p(\x)\mathrm{d}\x \\
& = \mathbb{E}_{\x}[(h(\x) - h^\ast(\x))^2] & + & \mathbb{E}_{\x}[\operatorname{Var}_{y}[y|\x]],
\end{align*}
donde hemos utilizado la caracterización de la varianza en términos de los momentos de primer y segundo orden.

El segundo sumando no depende de $h$ así que no afecta en la minimización.
El primer sumando es siempre no negativo y mide la distancia entre una hipótesis y $h^\ast$.
En $h^\ast$ se anula, luego $E_{\operatorname{out}}$ se minimizará en $h^\ast$.

\newpage

### Apartado d)

> ¿Cuál es el valor de $E_{\operatorname{out}}(h^\ast)$?

Aplicando la fórmula obtenida en el apartado anterior
$$E_{\operatorname{out}}(h^\ast) = \mathbb{E}_{\x}[(h^\ast(\x) - h^\ast(\x))^2] +  \mathbb{E}_{\x}[\operatorname{Var}_{y}[y|\x]] = \mathbb{E}_{\x}[\operatorname{Var}_{y}[y|\x]],$$
con lo que el valor de $E_{\operatorname{out}}(h^\ast)$ es la varianza media de $y$ condicionada a $\x$.

### Apartado e)

> Dar una interpretación, en términos de una muestra de datos, de la definición de la hipótesis óptima.

Con una muestra concreta de datos $\mathcal{D} = \{(\x_n,y_n)\}_{n = 1, \dots, N}$, aproximaríamos la hipótesis óptima dando a cada $\x$ la media de las $y_k$ tales que $\x_k = \x$, esto es, si dado $\x$ definimos $$I_\x := \{n \;:\; \x_n = \x\}$$ tendremos que podemos aproximar la hipótesis óptima con la función $$h_{\mathcal{D}}(\x) := \frac{1}{|I_\x|} \sum_{n \in I_\x} y_n.$$

Bajo condiciones de independencia e idéntica distribución sobre $\mathcal{D}$ tendremos que esta hipótesis convergerá a $h^\ast$ cuando $N \to \infty$, dando así una interpretación a $h^\ast$ en términos de $\mathcal{D}$.



*****

**BONUS 2**

> Una modificación del algoritmo perceptron denominada ADALINE, incorpora en la regla
> de adaptación una poderación sobre la cantidad de movimiento necesaria. En PLA se
> aplica $\w_{\operatorname{new}} = \w_{\operatorname{old}} +y_n \x_n$
> y en ADALINE se aplica la regla
> $\w_{\operatorname{new}} = \w_{\operatorname{old}} +\eta(y_n -w^T \x_n )\x_n$.
> Considerar la función de error $E_n(\w) = (\max(0, 1 - y_n \w^T \x_n))^2$.
> Argumentar que la regla de adaptación de ADALINE es equivalente a gradiente descendente estocástico (SGD)
> sobre $\frac{1}{N}\sum_{n=1}^N E_n(\w)$.

**Nota**: *No he hecho el ejercicio de Bonus 2*.

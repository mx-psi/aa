---
title: Cuestionario 1
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: false
toc-depth: 1
toc-title: Índice
header-includes:
- \usepackage{stmaryrd}
- \newcommand{\cov}{\operatorname{cov}}
- \newcommand{\x}{\mathbf{x}}
---

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
###  b) Decidir si un determinado índice del mercado de valores subirá o bajará dentro de un periodo de tiempo fijado.
### c) Hacer que un dron sea capaz de rodear un obstáculo.
### d) Determinar el estado anímico de una persona a partir de una foto de su cara.
### e) Determinar el ciclo óptimo para las luces de los semáforos en un cruce con mucho tráfico.

# Pregunta 2

>  ¿Cuales de los siguientes problemas son más adecuados para una aproximación por aprendizaje y cuales más 
> adecuados para una aproximación por diseño? Justificar la decisión.

### a) Determinar si un vertebrado es mamífero, reptil, ave, anfibio o pez.
### b) Determinar si se debe aplicar una campaña de vacunación contra una enfermedad.
### c) Determinar perfiles de consumidor en una cadena de supermercados.
### d) Determinar el estado anímico de una persona a partir de una foto de su cara.
### e) Determinar el ciclo óptimo para las luces de los semáforos en un cruce con mucho tráfico.


# Pregunta 3

> Construir un problema de aprendizaje desde datos para un problema de clasificación de
> fruta en una explotación agraria que produce mangos, papayas y guayabas. Identificar los
> siguientes elementos formales $\mathcal{X} , \mathcal{Y}, \mathcal{D}, f$ del problema. 
> Dar una descripción de los mismos que pueda ser usada por un computador. 
> ¿Considera que en este problema estamos ante un caso de etiquetas con ruido o sin ruido? 
> Justificar las respuestas.

# Pregunta 4

>  Suponga una matriz cuadrada $A$ que admita la descomposición $A = X^TX$ para alguna
> matriz $X$ de números reales. Establezca una relación entre los valores singulares de las
> matriz $A$ y los valores singulares de $X$.

Sea $X \in \mathcal{M}_{n\times m}(\mathbb{R})$. 
$X$ admite descomposición en valores singulares de la forma $$X = U D V^T$$ con 
$U,V$ matrices unitarias (en particular ortogonales) y
$D$ matriz diagonal.
Los elementos de la diagonal de $D$ serán los valores singulares de $X$.

Utilizando esta descomposición tenemos que $$A = XX^T = (U DV^T) (U DV^T)^T = U DV^TVDU^T = UD^2U^T,$$
donde hemos utilizado que $(A_1A_2)^T = A_2^TA_1^T$ y que la inversa de una matriz unitaria es su traspuesta.

Esta es una descomposición en valores singulares, ya que como $U$ es unitaria, también lo será $U^T$.
Por tanto tenemos que los valores singulares de $A$ serán los valores de la diagonal de $D^2$, es decir:
los valores singulares de $A$ serán **los cuadrados de los valores singulares de $X$**.

# Pregunta 5

> Sean  $\mathbf{x}$ e $\mathbf{y}$ dos vectores de características de dimensión $M \times 1$. La expresión
> $$\cov(\mathbf{x},\mathbf{y}) = \frac{1}{M}\sum_{i = 1}^M (x_i - \bar{x})(y_i - \bar{y})$$
> define la covarianza entre dichos vectores, donde $\bar{z}$ representa el valor medio de los elementos
> de $\mathbf{z}$. Considere ahora una matriz $X$ cuyas columnas representan vectores de características.
> La matriz de covarianzas asociada a la matriz $X = (\mathbf{x}_1 , \mathbf{x}_2 , \dots , \mathbf{x}_N)$ es el 
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
### b) Identifique la propiedad más relevante de dicha matriz en relación con regresión lineal.

# Pregunta 7

> La regla de adaptación de los pesos del Perceptron 
> ($\mathbf{w}_{\operatorname{new}} = \mathbf{w}_{\operatorname{old}} + y\mathbf{x}$) 
> tiene la interesante propiedad de que mueve el vector de pesos en la dirección adecuada 
> para clasificar $\mathbf{x}$ de forma correcta. 
> Suponga el vector de pesos $\mathbf{w}$ de un modelo y un dato $\mathbf{x}(t)$ mal clasificado
> respecto de dicho modelo. Probar matemáticamente que el movimiento de la regla de
> adaptación de pesos siempre produce un movimiento de $\mathbf{w}$ en la dirección correcta para
> clasificar bien $\mathbf{x}(t)$.

# Pregunta 8

>  Sea un problema probabilístico de clasificación binaria con etiquetas $\{0,1\}$, es decir
> $P(Y = 1) = h(\mathbf{x})$ y $P(Y = 0) = 1 - h(\mathbf{x})$, para una función $h$ dependiente de la muestra.

### Apartado a)

> Considere una muestra i.i.d. de tamaño $N$ $(\mathbf{x}_1 , \dots, \mathbf{x}_N)$. Mostrar que la función $h$ que maximiza la verosimilitud de la muestra es la misma que minimiza
> $$E_{\operatorname{in}}(\mathbf{w}) = \sum_{n=1}^N \llbracket y_n = 1\rrbracket \operatorname{ln}\frac{1}{h(\mathbf{x}_n)} + \llbracket y_n = 0\rrbracket \operatorname{ln}\frac{1}{1-h(\mathbf{x}_n)}$$
> donde $\llbracket\cdot\rrbracket$ vale 1 o 0 según que sea verdad o falso respectivamente la expresión en su interior.

### Apartado b)

>  Para el caso $h(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$ mostrar que minimizar el error de la muestra en el
>  apartado anterior es equivalente a minimizar el error muestral
> $$E_{\operatorname{in}}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^N \operatorname{ln}\left(1 + \exp(-y_n\mathbf{w}^T\mathbf{x}_n)\right)$$

Sea $n$ tal que $y_n = 1$.
El sumando en este caso vale 
\begin{align*}
\operatorname{ln}\left(\frac{1}{h(\x_n)}\right) & = \operatorname{ln}\left(\frac{1}{\sigma(\mathbf{w}^T \mathbf{x_n})}\right) \\
& =  \operatorname{ln}\left(\left(\frac{1}{1 + \exp(-\mathbf{w}^T \mathbf{x_n})}\right)^{-1}\right) \\
& = \operatorname{ln}\left(1 + \exp(-\mathbf{w}^T \mathbf{x_n})\right) \\
& = \operatorname{ln}\left(1 + \exp(-y_n\mathbf{w}^T \mathbf{x_n})\right),
\end{align*}
donde la última igualdad se da porque asumimos que $y_n = 1$.

TODO

# Pregunta 9

>  Derivar el error Ein para mostrar que en regresión logística se verifica:
> $$\nabla E_{\operatorname{in}}(\mathbf{w}) = - \frac{1}{N}\sum_{n=1}^N \frac{y_n \mathbf{x}_n}{1 + \exp(y_n\mathbf{w}^T\mathbf{x}_n)} = \frac{1}{N}\sum_{n=1}^N -y_n \mathbf{x}_n\sigma(-y_n\mathbf{w}^T\mathbf{x}_n)$$
> Argumentar sobre si un ejemplo mal clasificado contribuye al gradiente más que un ejemplo
> bien clasificado.

### Gradiente del error

Como en el ejercicio anterior, noto $\exp(x) = e^x$.

Calculamos la derivada con respecto de $\mathbf{w}_i$.
En primer lugar, como la derivada es un operador lineal tenemos que
$$\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i} = \frac{1}{N} \sum_{n=1}^N \frac{\partial}{\partial w_i}\left(\operatorname{log}(1 + \exp(-y_n\mathbf{w}^T\x_n))\right)$$

Para derivar esta expresión interior utilizamos la regla de la cadena y utilizando las derivadas del logaritmo y la exponencial tenemos que 
\begin{align*}
\frac{\partial}{\partial w_i}\left(\operatorname{log}(1 + \exp(-y_n\mathbf{w}^T\x_n))\right) & = \frac{\exp(-y_n\mathbf{w}^T\mathbf{x}_n)}{1 + \exp(-y_n\mathbf{w}^T\mathbf{x}_n)} \cdot \frac{\partial}{\partial w_i}\left(-y_n\mathbf{w}^T\mathbf{x}_n\right) \\ 
& = \sigma(-y_n\mathbf{w}^T\mathbf{x}_n) \cdot \frac{\partial}{\partial w_i}\left(-y_n\mathbf{w}^T\mathbf{x}_n\right)
\end{align*}

Para esta última expresión vemos que $$\mathbf{w}^T\mathbf{x}_n = \sum_{i=1}^M \mathbf{w}_i(\x_n)_i,$$
luego tenemos que $$\frac{\partial}{\partial w_i}\left(-y_n\mathbf{w}^T\mathbf{x}_n\right) = -y_n(\x_n)_i.$$

Juntando estas ecuaciones obtenemos 
$$\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i} = \frac{1}{N} \sum_{n=1}^N -y_n\sigma(-y_n\mathbf{w}^T\mathbf{x}_n)(\x_n)_i,$$
o expresándolo en términos del gradiente
$$\nabla E_{\operatorname{in}}(w) = \left(\frac{\partial E_{\operatorname{in}}(w)}{\partial w_i}\right)_{i=1\dots M} = \frac{1}{N} \sum_{n=1}^N  -y_n\sigma(-y_n\mathbf{w}^T\mathbf{x}_n)\x_n.$$

### Contribución al gradiente de ejemplos bien y mal clasificados

TODO

# Pregunta 10

> Definamos el error en un punto $(\mathbf{x}_n, y_n )$ por
> $$\mathbf{e}_n (\mathbf{w}) = \max(0, -y_n \mathbf{w}^T \mathbf{x}_n)$$
> Argumentar si con esta función de error el algoritmo PLA puede interpretarse como SGD
> sobre $\mathbf{e}_n$ con tasa de aprendizaje $\nu = 1$.


# BONUS 1

>  En regresión lineal con ruido en las etiquetas, el *error fuera de la muestra para una $h$ dada* puede expresarse como
> $$E_{\operatorname{out}}(h) = \mathbb{E}_{\mathbf{x},y}[(h(\mathbf{x}) - y)^2] = \iint (h(\mathbf{x}) - y)^2p(\mathbf{x}, y)\mathrm{d}\mathbf{x}\mathrm{d}y.$$

### Apartado a)

> Desarrollar la expresión y mostrar que
> $$E_{\operatorname{out}}(h) = \int \left( h(\mathbf{x})^2\int p(y|\mathbf{x})\mathrm{d}y - 2h(\mathbf{x})\int y \cdot p(y | \mathbf{x})\mathrm{d}y + \int y^2p(y|\mathbf{x})\mathrm{d}y \right)p(\mathbf{x})\mathrm{d}\mathbf{x}.$$

### Apartado b)

> El término entre paréntesis en Eout corresponde al desarrollo de la expresión
> $$\int (h(\mathbf{x})-y)^2p(y|\mathbf{x})\mathrm{d}y$$
> ¿Que mide este término para una $h$ dada?.

### Apartado c)

> El objetivo que se persigue en Regression Lineal es encontrar la función $h \in \mathcal{H}$ que
> minimiza $E_{\operatorname{out}}(h)$. Verificar que si la distribución de probabilidad $p(\mathbf{x}, y)$ con la que
> extraemos las muestras es conocida, entonces la hipótesis óptima $h^\ast$ que minimiza
> $E_{\operatorname{out}}(h)$ está dada por
> $$h^\ast(\mathbf{x}) = \mathbb{E}_y[y|\mathbf{x}] = \int y \cdot p(y|\mathbf{x})\mathrm{d}y.$$

TODO: qué

### Apartado d)

> ¿Cuál es el valor de $E_{\operatorname{out}}(h^\ast)$?

### Apartado e)

> Dar una interpretación, en términos de una muestra de datos, de la definición de la hipótesis óptima.

# BONUS 2

> Una modificación del algoritmo perceptron denominada ADALINE, incorpora en la regla
> de adaptación una poderación sobre la cantidad de movimiento necesaria. En PLA se
> aplica $\mathbf{w}_{\operatorname{new}} = \mathbf{w}_{\operatorname{old}} +y_n \mathbf{x}_n$
> y en ADALINE se aplica la regla
> $\mathbf{w}_{\operatorname{new}} = \mathbf{w}_{\operatorname{old}} +\eta(y_n -w^T \mathbf{x}_n )\mathbf{x}_n$.
> Considerar la función de error $E_n(\mathbf{w}) = (\max(0, 1 - y_n \mathbf{w}^T \mathbf{x}_n))^2$.
> Argumentar que la regla de adaptación de ADALINE es equivalente a gradiente descendente estocástico (SGD)
> sobre $\frac{1}{N}\sum_{n=1}^N E_n(\mathbf{w})$.


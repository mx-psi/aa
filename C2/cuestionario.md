---
title: Cuestionario 2
subtitle: Aprendizaje Automático
author: Pablo Baeyens Fernández
date: Curso 2018-2019
documentclass: scrartcl
toc: true
toc-depth: 1
toc-title: Índice
header-includes:
- \usepackage{stmaryrd}
- \usepackage{etoolbox}
- \AtBeginEnvironment{quote}{\itshape}
---

\newpage

# Pregunta 1

> Identificar de forma precisa dos condiciones imprescindibles para que un problema de
> predicción puede ser aproximado por inducción desde una muestra de datos. Justificar la
> respuesta usando los resultados teóricos estudiados.

La primera condición imprescindible es que las muestras deben ser **independientes e idénticamente distribuidas**, esto es, deben provenir de la misma distribución (de la que queremos predecir) y no deben tener dependencias entre sí. Esta condición es imprescindible para asegurar que podemos aplicar la desigualdad de Hoeffding, que nos proporciona la cota en el error.

La segunda condición es que **el tamaño de la muestra sea suficientemente grande**. 
La cota del error de la desigualdad de Hoeffding, dada por,
$$P[\mathcal{D} : |\mu - \eta| > \varepsilon] \leq 2e^{-2\varepsilon^2 N}$$
es decreciente en función del tamaño de la muestra $N$.
Si la muestra es suficientemente grande podremos alcanzar el valor de precisión deseado con la probabilidad tan alta como queramos.


# Pregunta 2

> El jefe de investigación de una empresa con mucha experiencia en problemas de predicción de
> datos tras analizar los resultados de los muchos algoritmos de aprendizaje usados sobre todos
> los problemas en los que la empresa ha trabajado a lo largo de su muy dilatada existencia,
> decide que para facilitar el mantenimiento del código de la empresa van a seleccionar un
> único algoritmo y una única clase de funciones con la que aproximar todas las soluciones a
> sus problemas presentes y futuros. ¿Considera que dicha decisión es correcta y beneficiará
> a la empresa? Argumentar la respuesta usando los resultados teóricos estudiados.

Considero que la decisión es **incorrecta**.

El teorema de *No-Free-Lunch* nos indica que para cualquier algoritmo (en particular el escogido por el jefe de investigación) existirá una distribución de probabilidad para la cuál fallará. Por tanto, si nos encontramos con tal distribución el algoritmo no dará buenos resultados y no será beneficioso para la empresa.

Además, cualquier algoritmo que haya escogido el jefe de investigación será igualmente bueno en media sobre todas las distribuciones. El único caso en el que el jefe de investigación podría estar tomando una decisión correcta es si tiene conocimiento específico del dominio al que pertenecen los problemas a los que se enfrentará la empresa en el futuro, en cuyo caso podría afirmar que un algoritmo puede tener mejores resultados en esta clase específica.


# Pregunta 3

> ¿Que se entiende por una solución PAC a un problema de aprendizaje? Identificar el porqué
> de la incertidumbre e imprecisión.

Una clase de hipótesis $\mathcal{H}$ es aprendible de forma PAC si existe una función 
$m_{\mathcal{H}}:]0,1[^2 \to \mathbb{N}$
y un algoritmo de aprendizaje tal que 
para cada 

- imprecisión $\varepsilon \in ]0,1[$, 
- incertidumbre $\delta \in ]0,1[$ y
- distribución $\mathcal{P}$ sobre $\mathcal{X}$,

si ejecutamos el algoritmo con un número de muestras $N$ independientes idénticamente distribuidas mayor a $m_{\mathcal{H}}(\varepsilon, \delta)_{}$ (provenientes de la distribución $\mathcal{P}$), tendremos que el algoritmo devuelve una hipótesis $h \in \mathcal{H}$ tal que
$$P[E_{\operatorname{in},\mathcal{P}}(h) - \min_{h' \in \mathcal{H}}(E_{\operatorname{in},\mathcal{P}}(h')) \leq \varepsilon] \geq 1 - \delta,$$
donde la probabilidad se mide sobre la distribución de las $N$ muestras i.i.d.s.

Esta solución $h$ será una solución *PAC*, del inglés *Probably Approximately Correct* (probablemente aproximadamente correcta), ya que la diferencia de su error empírico con respecto a la mejor hipótesis posible de la clase es menor que una cierta imprecisión $\varepsilon$ con una incertidumbre de como mucho $\delta$.

La incertidumbre y precisión vienen de que la solución sea *probablemente aproximadamente* correcta: con alta probabilidad, al menos $1-\delta$, será aproximadamente correcta, con imprecisión de como mucho $\varepsilon$.

No podemos descartar estos parámetros (esto es, no podemos tener certeza y precisión absoluta fijando $\varepsilon = 0, \delta = 0$) ya que siempre existe la posibilidad de que las muestras no sean representativas (sólo tenemos una cota probable del error usando la desigualdad de Hoeffding).

# Pregunta 4

> Suponga un conjunto de datos $\mathcal{D}$ de 25 ejemplos extraidos de una funcion desconocida
> $f : \mathcal{X} \to \mathcal{Y}$, donde $\mathcal{X} = \mathbb{R}$ 
> e $\mathcal{Y} = \{-1, +1\}$. Para aprender $f$ usamos un conjunto simple
> de hipótesis $\mathcal{H} = \{h_1 , h_2\}$ donde $h_1$ es la función constante igual a $+1$ y $h_2$ la función
> constante igual a $-1$. Consideramos dos algoritmos de aprendizaje, S(smart) y C(crazy). S
> elige la hipótesis que mejor ajusta los datos y C elige deliberadamente la otra hipótesis.

## a) ¿Puede S producir una hipótesis que garantice mejor comportamiento que la aleatoria sobre cualquier punto fuera de la muestra?
Justificar la respuesta


# Pregunta 5

> Con el mismo enunciado de la pregunta.4:

## a) Asumir desde ahora que todos los ejemplos en $\mathcal{D}$ tienen $y_n = +1$. ¿Es posible que la hipotesis que produce C sea mejor que la hipótesis que produce S?

Justificar la respuesta



# Pregunta 6

> Considere la cota para la probabilidad de la hipótesis solución g de un problema de
> aprendizaje, a partir de la desigualdad generalizada de Hoeffding para una clase finita de
> hipótesis,
> $$P[|E_{\operatorname{out}}(g) - E_{\operatorname{in}} (g)| > \varepsilon) < \delta$$

## a) ¿Cuál es el algoritmo de aprendizaje que se usa para elegir g?
## b) Si elegimos g de forma aleatoria ¿seguiría verificando la desigualdad?
## c) ¿Depende g del algoritmo usado?
## d) Es una cota ajustada o una cota laxa?

# Pregunta 7

> ¿Por qué la desigualdad de Hoeffding definida para clases $\mathcal{H}$ de una única función no es
> aplicable de forma directa cuando el número de hipótesis de $\mathcal{H}$ es mayor de 1?

No podemos aplicar la desigualdad de forma directa porque la hipótesis debe estar fijada de antemano.

# Pregunta 8

> Si queremos mostrar que $k^\ast$ es un punto de ruptura para una clase de funciones H cuales
> de las siguientes afirmaciones nos servirían para ello:

## a) Mostrar que existe un conjunto de $k^\ast$ puntos $x_1, \dots, x_{k^\ast}$ que $\mathcal{H}$ puede separar («shatter»).

## b) Mostrar que $\mathcal{H}$ puede separar cualquier conjunto de $k^{\ast}$ puntos.

## c) Mostrar un conjunto de $k^\ast$ puntos $x_1, \dots, x_{k^\ast}$ que $\mathcal{H}$ no puede separar
## d) Mostrar que $\mathcal{H}$ no puede separar ningún conjunto de $k^\ast$ puntos
## e) Mostrar que $m_{\mathcal{H}}(k) = 2^{k^\ast}$


# Pregunta 9

> Para un conjunto $\mathcal{H}$ con $d_{\operatorname{VC}} = 10$, ¿qué tamaño muestral se necesita (según la cota 
> de generalización) para tener un 95 % de confianza ($\delta$) de que el error de generalización ($\varepsilon$)
> sea como mucho 0.05?

Utilizando la cota del error de generalización tenemos que debemos encontrar, para $\delta = 0.05$ y $\varepsilon < 0.05$ un $N$ tal que 
$$N \geq \frac{8}{\varepsilon^2}\log\left(\frac{4((2N)^{d_{\operatorname{VC}}} + 1)}{\delta}\right) > \frac{8}{400^{-1}} \log\left(\frac{4((2N)^{10} + 1)}{0.05}\right) = 3200 \cdot \log\left(80 \cdot ((2N)^{10} + 1)\right).$$

Esta es una desigualdad en forma implícita, por lo que utilizamos un programa de cálculo numérico para obtener una solución.
En particular, en Mathematica ejecutamos 
```mathematica
Reduce[x >= 3200 Log[80 + 81920 x^10], x]
```
La solución obtenida es que $N > \lceil 452956.8 \rceil = 452957$.

Por tanto, tenemos que necesitamos un tamaño muestral de al menos 452957 muestras independientes idénticamente distribuidas para tener una solución con error de como mucho $\varepsilon = 0.05$ con certeza $1 - \delta = 0.95$.


# Pregunta 10

> Considere que le dan una nuestra de tamaño $N$ de datos etiquetados $\{-1, +1\}$ y le piden
> que encuentre la función que mejor ajuste dichos datos. Dado que desconoce la verdadera
> función $f$, discuta los pros y contras de aplicar los principios de inducción ERM y SRM
> para lograr el objetivo. Valore las consecuencias de aplicar cada uno de ellos.

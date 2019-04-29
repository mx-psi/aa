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


# Pregunta 2

>  El jefe de investigación de una empresa con mucha experiencia en problemas de predicción de
> datos tras analizar los resultados de los muchos algoritmos de aprendizaje usados sobre todos
> los problemas en los que la empresa ha trabajado a lo largo de su muy dilatada existencia,
> decide que para facilitar el mantenimiento del código de la empresa van a seleccionar un
> único algoritmo y una única clase de funciones con la que aproximar todas las soluciones a
> sus problemas presentes y futuros. ¿Considera que dicha decisión es correcta y beneficiará
> a la empresa? Argumentar la respuesta usando los resultados teóricos estudiados.


# Pregunta 3

> ¿Que se entiende por una solución PAC a un problema de aprendizaje? Identificar el porqué
> de la incertidumbre e imprecisión.

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


# Pregunta 10

> Considere que le dan una nuestra de tamaño $N$ de datos etiquetados $\{-1, +1\}$ y le piden
> que encuentre la función que mejor ajuste dichos datos. Dado que desconoce la verdadera
> función $f$, discuta los pros y contras de aplicar los principios de inducción ERM y SRM
> para lograr el objetivo. Valore las consecuencias de aplicar cada uno de ellos.

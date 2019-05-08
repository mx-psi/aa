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

La primera condición imprescindible es que la muestra de datos **provengan de la distribución que queremos aproximar** por inducción. Si este no es el caso, por ejemplo, si la distribución ha variado desde la obtencion de los datos, no podremos aprender nada.

La segunda condición es que las muestras deben ser **independientes e idénticamente distribuidas**, esto es, deben provenir de la misma distribución y no deben tener dependencias entre sí. Esta condición es imprescindible para asegurar que podemos aplicar la desigualdad de Hoeffding, que nos proporciona la cota en el error.

Hay otras condiciones, como tener un tamaño de la muestra suficientemente grande que nos permiten que la aproximación sea más o menos precisa, pero si no se cumplen estas condiciones imprescindibles no podremos aproximar el problema con ninguna precisión. 

<!-- La segunda condición es que **el tamaño de la muestra sea suficientemente grande**.  -->
<!-- La cota del error de la desigualdad de Hoeffding, dada por, -->
<!-- $$P[\mathcal{D} : |\mu - \eta| > \varepsilon] \leq 2e^{-2\varepsilon^2 N}$$ -->
<!-- es decreciente en función del tamaño de la muestra $N$. -->
<!-- Si la muestra es suficientemente grande podremos alcanzar el valor de precisión deseado con la probabilidad tan alta como queramos. -->


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


\newpage

# Pregunta 4

> Suponga un conjunto de datos $\mathcal{D}$ de 25 ejemplos extraidos de una función desconocida
> $f : \mathcal{X} \to \mathcal{Y}$, donde $\mathcal{X} = \mathbb{R}$ 
> e $\mathcal{Y} = \{-1, +1\}$. Para aprender $f$ usamos un conjunto simple
> de hipótesis $\mathcal{H} = \{h_1 , h_2\}$ donde $h_1$ es la función constante igual a $+1$ y $h_2$ la función
> constante igual a $-1$. Consideramos dos algoritmos de aprendizaje, S(smart) y C(crazy). S
> elige la hipótesis que mejor ajusta los datos y C elige deliberadamente la otra hipótesis.

## a) ¿Puede S producir una hipótesis que garantice mejor comportamiento que la aleatoria sobre cualquier punto fuera de la muestra?

**No**, no podemos *garantizar* el mejor comportamiento, ya que el conjunto de datos $\mathcal{D}$ se ha producido mediante un proceso aleatorio por lo que es posible que los datos no sean representativos de la distribución total sobre $\mathcal{X}$.

Podemos dar una cota del error mediante la desigualdad generalizada de Hoeffding al encontrarnos con una clase finita de hipótesis, pero esta cota no está garantizada: sólo es cierta aceptando una cierta incertidumbre prefijada, que dependerá de la precisión $\varepsilon$ y del tamaño de la muestra (25 en este caso).


# Pregunta 5

> Con el mismo enunciado de la pregunta.4:

## a) Asumir desde ahora que todos los ejemplos en $\mathcal{D}$ tienen $y_n = +1$. ¿Es posible que la hipotesis que produce C sea mejor que la hipótesis que produce S?

**Sí**, cabe la posibilidad de que la hipótesis producida por $C$ sea mejor que la producida por $S$, 
ya que de nuevo nos hallamos ante un proceso aleatorio en el que los datos podrían no ser representativos.

Sin embargo, si los datos han sido generados bajo las condiciones de la desigualdad generalizada de Hoeffding la probabilidad de que los errores empíricos de ambas hipótesis sean lejanos al error real es muy baja.
Estas estimaciones nos dicen que S tiene un error empírico mucho menor que $C$ (que clasificaría incorrectamente todos los ejemplos de la muestra), por lo que, con alta probabilidad la hipótesis que produce C sería pero que la que produce S.


# Pregunta 6

> Considere la cota para la probabilidad de la hipótesis solución g de un problema de
> aprendizaje, a partir de la desigualdad generalizada de Hoeffding para una clase finita de
> hipótesis,
> $$P[|E_{\operatorname{out}}(g) - E_{\operatorname{in}} (g)| > \varepsilon) < \delta$$

## a) ¿Cuál es el algoritmo de aprendizaje que se usa para elegir $g$?

Seguimos el criterio ERM, es decir, escogemos de entre todas las hipótesis de la clase $\mathcal{H}$, aquella $g$ que minimice $E_{\operatorname{in}}$, ya que sabemos que, con probabilidad $1 - \delta$, la distancia entre este error empírico y el error real será menor que la precisión fijada $\varepsilon$.

## b) Si elegimos $g$ de forma aleatoria ¿seguiría verificando la desigualdad?

Sí, como nos hallamos en una clase de hipótesis finita la cota de probabilidad a partir de la desigualdad (generalizada) de Hoeffding se aplica de forma uniforme a todas las hipótesis de la clase. 

Es decir, el error empírico de cualquier hipótesis $g$ es una estimación del error real de $g$ (bajo las condiciones de la desigualdad descritas en el apartado anterior).

## c) ¿Depende $g$ del algoritmo usado?

Sí, $g$ puede depender del algoritmo utilizado, por ejemplo si utilizamos otro criterio que no sea el de minimización de riesgo empírico (sino que añadimos alguna medida de regularización).

## d) Es una cota ajustada o una cota laxa?

La cota del error que proporciona la desigualdad generalizada de Hoeffding para una clase finita de hipótesis es una cota **laxa**, ya que la desigualdad se obtiene mediante la subaditividad de la probabilidad, esto es, si $E_1, \dots, E_n$ son eventos, se cumple
$$P\left[\bigcup_{i = 1}^n E_i \right] \leq \sum_{i = 1}^n P[E_i].$$

Pero esta desigualdad no es muy ajustada, ya que si los eventos aleatorios $E_1,\dots, E_n$ tienen intersección probable estaremos sumando de más. Por tanto no obtenemos mucha información de esta cota del error (más allá de su comportamiento cuando el tamaño de la muestra tiende a infinito).

# Pregunta 7

> ¿Por qué la desigualdad de Hoeffding definida para clases $\mathcal{H}$ de una única función no es
> aplicable de forma directa cuando el número de hipótesis de $\mathcal{H}$ es mayor de 1?

No podemos aplicar la desigualdad de forma directa porque la hipótesis a la que se la aplicamos debe estar fijada de antemano y en el caso de tener una clase de hipótesis con $|\mathcal{H}| > 1$ la hipótesis no está fijada sino que se escoge por el algoritmo de aprendizaje en función de los datos. 

Si pudiéramos cambiar la hipótesis después de observar el conjunto de datos no se cumplirían las precondiciones necesarias para la desigualdad y por tanto debemos ajustarla para que esto se cumpla.

En el caso de una clase de hipótesis finita podemos aplicar la subaditividad de la función de probabilidad para ver que una posible cota es $2|\mathcal{H}|e^{-2\varepsilon^2N}$ donde $\varepsilon$ es la precisión y $N$ el número de muestras. Una clase de hipótesis infinita requerirá del uso de la dimensión VC.

# Pregunta 8

> Si queremos mostrar que $k^\ast$ es un punto de ruptura para una clase de funciones H cuales
> de las siguientes afirmaciones nos servirían para ello:

Un punto de ruptura para la clase de hipótesis $\mathcal{H}$ es un entero $k^\ast$ tal que no podemos separar cualquier conjunto de $k^\ast$ puntos.

## a) Mostrar que existe un conjunto de $k^\ast$ puntos $x_1, \dots, x_{k^\ast}$ que $\mathcal{H}$ puede separar («shatter»).

**No** nos serviría, ya que un punto de ruptura es aquel para el que todo conjunto de ese cardinal **no** puede separarse.

Por ejemplo los perceptrones 2D pueden separar cualquier conjunto de $k^\ast = 2$ puntos, pero este no es su punto de ruptura.


## b) Mostrar que $\mathcal{H}$ puede separar cualquier conjunto de $k^{\ast}$ puntos.

**No** nos serviría, ya que lo que queremos es encontrar un punto en el que no podamos separar ningún conjunto de ese cardinal.
De nuevo el ejemplo de los perceptrones 2D sirve.

## c) Mostrar un conjunto de $k^\ast$ puntos $x_1, \dots, x_{k^\ast}$ que $\mathcal{H}$ no puede separar

**No** nos serviría para mostrar que es un punto de ruptura, tenemos que no poder separar ningún conjunto de ese tamaño (y no sólo uno).

## d) Mostrar que $\mathcal{H}$ no puede separar ningún conjunto de $k^\ast$ puntos

**Sí** nos serviría, esta es exactamente la condición que buscamos.

Por ejemplo los «rayos positivos» en $\mathbb{R}$ (esto es, las hipótesis de la forma $\operatorname{sgn}(x-w_0)$) no pueden separar ningún conjunto de 4 puntos, pero tampoco ninguno de 3.

## e) Mostrar que $m_{\mathcal{H}}(k) = 2^{k^\ast}$

**No** nos serviría, esto indica que hay algún conjunto de $k$ puntos que podemos separar, y lo que buscamos es no poder separar ninguno.

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

En el caso de **ERM** (del inglés *Empirical Risk Minimization*) fijamos una clase de hipótesis $\mathcal{H}$ y tomamos la hipótesis que minimice el *error empírico* $E_{\operatorname{in}}$ sobre la muestra que tenemos.

Si la muestra es suficientemente grande, independiente idénticamente distribuida y proveniente de la distribución que queremos aprender y la clase de hipótesis tiene dimensión VC finita podemos asegurar con alta probabilidad que esta hipótesis tiene un error real $E_{\operatorname{out}}$ cercano al mínimo posible.

El conjunto de clases de hipótesis que podemos considerar con este criterio es reducido ya que la clase de hipótesis debe tener dimensión VC finita. A cambio minimizamos el error real sin restricciones.

Este criterio es útil bajo las condiciones anteriormente mencionadas. 
Bajo estas condiciones su principal ventaja es su simplicidad.
pero si el tamaño de la muestra es pequeño o consideramos que no podemos escoger ninguna clase de hipótesis suficientemente diversa podría no ser el ideal.

Por otro lado, en el principio de inducción **SRM** (del inglés *Structural Risk Minimization*) tenemos una clase de hipótesis que puede definirse como unión de una cantidad numerable de subclases $$\mathcal{H} = \bigcup_{n \in \mathbb{N}} \mathcal{H}_n$$ de tal forma que cada subclase tenga dimensión VC finita.

Debemos escoger una hipótesis que minimice la suma del error empírico $E_{\operatorname{in}}$ y una penalización por la complejidad del modelo $\Omega(\mathcal{H}_n)$. De esta forma minimizamos a la vez la complejidad de la hipótesis elegida (minimizando la penalización) y el error empírico de la muestra, para lo que podemos fijar la complejidad del modelo y minimizar el error o fijar el error y minimizar la complejidad.
Bajo ciertas condiciones esto minimizará el error real.

Este criterio tiene como ventaja que las clases de hipótesis consideradas pueden ser más generales; el ejemplo más sencillo es considerar los polinomios de cualquier grado. Su desventaja es su mayor complejidad a la hora de implementarse y el hecho de que la muestra necesaria depende de la función.

# Bonus 

## Pregunta 1

> Considere que le dan un conjunto de datos yque tras echarles un vistazo observa
> que son separables linealmente. Por tanto ajusta un modelo perceptron y obtiene un error
> zero sobre los datos de aprendizaje. Entonces desea obtener una cota de generalización
> para lo cual mira la dimension de VC del modelo ajustado y ve que es $d+1$. Por tanto usa
> esa cota para obtener una cota del error del modelo.

### a) ¿Hay algún problema con la cota elegida? ¿es correcta?

**Sí**, el problema está en que hemos observado los datos antes de ajustar con un modelo concreto, por tanto la cota podría no ser correcta. 

Al haber decidido el modelo después de la observación no estamos bajo los supuestos del modelo PAC y por tanto no podemos asegurar que la cota sea correcta.

### b) ¿Conocemos la cota de VC para el modelo que hemos usado realmente?

**No**, sólo conocemos la cota de VC para el modelo si hubiéramos seguido los supuestos de la teoría (esto es, si no hubiéramos observado los datos para elegir la hipótesis).

Como hemos observado los datos el conjunto de hipótesis es potencialmente mucho más grande, ya que la clase de hipótesis que realmente hemos considerado contiene muchas más hipótesis (todas las que conocemos y podríamos aplicar).

### c) Si la cota no fuera correcta, ¿cúal deberíamos haber usado?

No hay respuesta en la teoría de aprendizaje PAC para este caso, de hecho, la dimensión VC de la clase de hipótesis posibles es potencialmente infinita, con lo que no habría ninguna cota que pudiéramos utilizar.

\newpage

## Pregunta 2

> Suponga un conjunto de datos y extrae de él 100 muestras que no serán usados
> en entrenamiento sino que serán usados para seleccionar una de las tres hipótesis $g_1, g_2, g_3$
> producidas por tres algoritmos diferentes que serán entrenados con el resto de los datos.
> Cada algoritmo trabaja con una clase diferente $\mathcal{H}$ de 500 funciones. Queremos caracterizar
> la precisión de la estimación de $E_{\operatorname{out}}(g)$ sobre la hipótesis final seleccionada a partir de las
> 100 muestras.

### a) ¿Qué valor de $M$ debería de usarse en la expresión $2Me^{2-N\varepsilon^2}$ de la desigualdad de Hoeffding generalizada?

El número de hipótesis que hemos considerado de forma efectiva es un total de $500 \cdot 3 = 1500$ hipótesis, ya que estamos considerando una clase diferente de $500$ hipótesis para cada una de las hipótesis $g_1, g_2$ y $g_3$.

Por tanto debemos tomar $M = 1500$ en la cota de la desigualdad de Hoeffding generalizada.

### b) ¿Compare el nivel de contaminación de estas 100 muestras con el caso donde estas muestras hubieran participado en entrenamiento en lugar del proceso de selección?

Si usamos las muestras sólo en entrenamiento o sólo en selección el resultado es el mismo: la cota es independiente del algoritmo utilizado, y el algoritmo que utilizamos incluye tanto el entrenamiento como en la selección.

En cambio, si reutilizamos estas 100 muestras tanto en el proceso de entrenamiento como en el proceso de selección estas muestras no contarían para el $N$ que utilizamos en la cota del error, por lo que el error real podría ser más grande. Además, correríamos potencialmente el riesgo de sobreajuste.

## Pregunta 3

**Nota**: *No he hecho la pregunta BONUS 3*.

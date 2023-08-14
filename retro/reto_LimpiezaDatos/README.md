# Análisis de la Base de Datos del Titanic

## Introducción

El reto del curso consiste en la utilización del método CRISP-DM (Cross Industry Standard Process for Data Mining) para generar una solución que integre ciencia de datos e inteligencia artificial utilizando herramientas computacionales de vanguardia. El objetivo es obtener una base de datos confiable que permita analizar y predecir una variable. En este caso, se aborda el reto de predecir si una persona que estuvo a bordo del Titanic sobrevivió o no.

## Etapa 1: Comprensión del Negocio

Para abordar el problema, se comenzó identificando el desafío, los requisitos y las herramientas necesarias. Se descargó una base de datos de pasajeros del Titanic con diversas características, como nombre, número de ticket, sexo, edad, entre otros. Se eligió Python como lenguaje de programación, utilizando Google Colab para aprovechar las capacidades de matemáticas e inteligencia artificial. 

La primera fase para predecir la supervivencia de los pasajeros del Titanic implicó limpiar la base de datos, eliminando variables irrelevantes para el análisis de sobrevivencia, como ID, Nombre, Parch, Ticket y cabina.

## Limpieza de Datos

Para realizar un análisis efectivo y una limpieza adecuada de los datos, se utilizaron las siguientes librerías de Python: Pandas, Numpy y Matplotlib.

Se tomaron decisiones de limpieza, como eliminar las variables de ID, Nombre, Parch, Ticket y Cabina, ya que se consideraron no relevantes para el análisis de supervivencia en situaciones de emergencia.

```python
data_titanic = data_titanic.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
```

Después, decidimos cambiar las variables de sexo a ceros y unos, debido a que por naturaleza el sexo es binario, y nos fué más práctico para realizar el análizis debido a que es más fácil trabajar con ints que con strings, además decidimos reemplazar la variable “Embarked”, de ‘S’, ‘C’ Y ‘Q’ a 0,1,2 respectivamente.

Además utilizamos la función de dropnan() para quitar los datos no definidos de nuestra base, puesto que estos no nos aportarán información de las variables a analizar. Así mismo utilizamos la función dropnull() para remover datos nulos en nuestra base .

A partir de aquí partimos de la limpieza de datos para empezar a trabajar con ellos y encontrar estadísticas que nos permitan hacer un análisis de los mismos:

Primero hicimos un conteo y porcentajes de las categorías de sexo, embarcación, hombres sobrevimientes, hombres fallecidos, mujeres sobrevivientes, mujeres fallecidas. Así como un análisis de cuartiles y quintiles de las frecuencias de las edades de los pasajeros, señalando que las edades entre 20 y 40 eran las más frecuentes. Por último realizamos un análisis de correlación para identificar que variables están más relacionadas con otras y con sobrevivir. Encontramos que las variables más correlacionadas con la sobrevivencia es el sexo y el pasaje que se pagó.

## Conclusión 

Encontramos en nuestra limpieza y análisis que hay una fuerte correlación entre el sexo y la sobrevivencia, además de una correlación entre el pasaje y la sobrevivencia. Esto es de esperarse dado que en emergencias, las personas más cercanas a las salidas son más propensas a sobrevivir, en este caso, la clase que pagó más, estaba en un lugar más cercano a los botes salvavidas, así como también se procuraron mujeres y niños en esta situación.


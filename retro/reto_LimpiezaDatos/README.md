# Análisis de la Base de Datos del Titanic

## Introducción

El reto del curso consiste en la utilización del método CRISP-DM (Cross Industry Standard Process for Data Mining) para generar una solución que integre ciencia de datos e inteligencia artificial utilizando herramientas computacionales de vanguardia. Esto implica la obtención de una base de datos confiable que nos permita hacer un análisis de los mismos, con la finalidad de predecir una variable. En este caso, se encuentra con el reto de predecir si una persona que atendió al Titanic sobrevivió o no. 

## Etapa 1: Comprensión del Negocio

Lo primero que se realizó para empezar a solucionar el problema, fue identificar el problema, así como los requerimientos que tiene el reto y herramientas que utilizaremos para completar los requerimientos.

En primera instancia fue necesario descargar la base de datos de los pasajeros del Titanic con sus principales características, como lo es nombre, número de ticket, sexo, edad, etc. 

Lo segundo fue utilizar una plataforma que permite utilizar herramientas de matemáticas, así como inteligencia artificial para darnos una mejor oportunidad de resolver el problema, así como poder trabajar en conjunto. Para esto se decidió utilizar el lenguaje de programación Python, el cuál desarrollamos en la plataforma de Google Colab, por medio de un tipo de archivo llamado Jupiter Notebook.

También se identificó que el primer paso para generar una predicción del deceso o sobrevivencia de los pasajeros del Titanic es la limpieza de la base de datos, esto debido a que se tienen variables que no aportan información útil acerca del evento que ocasionó la muerte de muchos (en específico quién entró a las barcas de emergencia y quién no).

## Limpieza de Datos

Para realizar un análisis efectivo y una limpieza adecuada de los datos, se utilizaron las siguientes librerías de Python: Pandas, Numpy y Matplotlib.

Se tomaron decisiones de limpieza, como eliminar las variables de ID, Nombre, Parch, Ticket y Cabina, ya que se consideraron no relevantes para el análisis de supervivencia en situaciones de emergencia esto con la hipótesis de que lo más determinante para saber la supervivencia es la edad, sexo y clase de los pasajeros por lo que se conoce de la prioridad que se les dio a mujeres y niños para subir a los botes salvavidas. El número de Tickets se juzgó atípico ya que solo describe un folio que lleva la cuenta de boletos vendidos y no tiene relación con la supervivencia de un pasajero. 

```python
data_titanic = data_titanic.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
```

Después, se decidió cambiar las variables categóricas de sexo (“sex”) y embarcamiento (“embarked”) a variables numéricas de este modo se puede realizar una relación matemática de dichas variables al ser números y no caracteres, así que se remplazó el sexo masculino por un 0 y el femenino por un 1 y la variable de embarcamiento, de ‘S’, ‘C’ y ‘Q’ a 0,1,2 respectivamente.

Además se utilizó la función de dropnan() para quitar los datos no definidos de la base, puesto que estos no aportan información de las variables a analizar. Así mismo, gracias a la función dropnull() se removieron las filas que contuvieran datos nulos, esto para evitar el ruido que estas variables pueden provocar en el análisis.

A partir de aquí partimos de la limpieza de datos para empezar a trabajar con ellos y encontrar estadísticas que nos permitan hacer un análisis de los mismos:

Primero se realizó un conteo y porcentajes de las categorías de sexo, embarcación, hombres sobrevimientes, hombres fallecidos, mujeres sobrevivientes, mujeres fallecidas. Así como un análisis de cuartiles y quintiles de las frecuencias de las edades de los pasajeros, señalando que las edades entre 20 y 40 eran las más frecuentes. Por último se hizo un análisis de correlación para identificar qué variables están más relacionadas con otras y con sobrevivir. Encontramos que las variables más correlacionadas con la sobrevivencia es el sexo y el pasaje que se pagó. 

La tabla de correlación muestra una relación importante (cercana al 1) entre el sexo y la clase con  la sobrevivencia. Por otro lado, en cuanto a la edad, no se ve la relación fuerte esperada al tener una relación del 9% aproximadamente.

## Conclusión 

Se encontrón en la limpieza y análisis que hay una fuerte correlación entre el sexo y la sobrevivencia, además de una correlación entre el pasaje y la sobrevivencia. Esto es de esperarse dado que en emergencias, las personas más cercanas a las salidas son más propensas a sobrevivir, en este caso, la clase que pagó más, estaba en un lugar más cercano a los botes salvavidas, así como también se procuraron mujeres y niños en esta situación.

En lo que respecta a la edad, el hecho de no mostrar relación fuerte con la sobrevivencia fue desconcertante, se sospecha que la supresión de los pasajeros con datos nulos pudo provocar resultados alejados de la realidad (falsos). Por ello, se realizará otro análisis en el cual se dejarán los datos de los pasajeros eliminados en esta limpieza, agregando una variable que aproxime las edades de dichos pasajeros y ver la nueva correlación.



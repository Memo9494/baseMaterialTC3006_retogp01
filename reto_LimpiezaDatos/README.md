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

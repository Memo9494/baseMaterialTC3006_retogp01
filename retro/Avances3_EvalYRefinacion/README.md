# Evaluación y Refinamiento de modelo

## Introducción

En este reporte será detallado nuestro proceso para separar la base de datos en dos partes distintas, una para el entrenamiento de nuestro modelo y una para hacer las pruebas. También se justificarán las modificaciones hechas a la base de datos y las variables seleccionadas. Finalmente compararemos los resulatados obtenidos por los diferentes modelos y los cambios que hicimos para refinarlos y mejorar su rendimiento.

## Selección de variables

Para elegir qué variables serían utilizadas para el modelo se generó un diagrama de correlación, con el cual pudimos ver qué variables estaban más ligadas a la supervivencia, con este se seleccionaron las siguientes variables: 'Age', 'Sex', 'Pclass', 'Embarked' y 'Fare'. 'Survived' fue designada como nuestra variable de respuesta. Un conflicto con el que nos encontramos fueron los datos faltantes de la variable de edad, de todas las variables que seleccionamos solamente 'Age' tenía una alta cantidad de datos faltantes. Consideramos esta como una variable importante así que no sabíamos de qué forma proceder. Finalmente decidimos lo siguiente, utilizamos dos copias de nuestra base de datos, a una se le eliminaron todos los valores nulos, en la otra se preservaron estos datos faltantes, utilizamos los mismos modelos con estas dos bases para comparar los resultados y ver cuál nos daba mejores predicciones.

## Separación de los datos

Para dividir nuestra base de datos en dos se uso la función 'train_test_split()', proveniente de la librería 'sklearn'. Esta función nos permite hacer esta división facilmente y siguiendo los parámetros deseados. En nuestro caso decidimos usar un ratio de 80-20 en nuestro split, en otras palabras, el 80% de nuestros datos fueron utilizados para entrenar el modelo y los restantes para verificar los resultados de sus predicciones. También se utilizó el parámetro 'stratify' de la función para asegurarse que nuestros dos datasets tuvieran aproximadamente el mismo ratio de sobrevivientes y así tener resultados más representativos.

## Implementación de modelos

Primeramente implementamos una variedad de modelos, provenientes de la librería sklearn, con el objetivo de tener una idea general del rendimiento de cada modelo y de ahí seleccionar los mejores para hacer un ajuste de híper parámetros y conseguir sus mejores resultados. En esta primera etapa se usaron los siguientes modelos:

- K-Neighbors Classifier
- SVC (Support Vector Classifier)
- Logistic Regression
- Decision Tree Classifier
- Gaussian NB
- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Multi-layer Perceptron Classifier (MLP)

De estos nueve modelos los seleccionados para ser mejorados fueron Gradient Boosting, MLP, y Decision Tree, adicionalmente se añadió el modelo Extreme Gradient Boosting de la librería xgboost ya que es una versión más poderosa del gradient boosting.

La exactitud de cada modelo ates de hacer cambios se puede ver a continuación:

|  | Eliminando los valores nulos  | Preservando los valores nulos |
| ------------- | ------------- | ------------- |
| Gradient Boosting  | 0.78 | 0.80 |
| Extreme Gradient Boosting  | 0.80  | 0.81  |
| Multi-Layer Perceptron  | 0.74  | 0.78  |
| Decision Tree | 0.71  | 0.81  |

Una vez hecho el ajuste de híper parámetros podemos ver las siguientes mejoras:

|  | Eliminando los valores nulos  | Preservando los valores nulos |
| ------------- | ------------- | ------------- |
| Gradient Boosting  | 0.82 | 0.78 |
| Extreme Gradient Boosting  | 0.84  | 0.81  |
| Multi-Layer Perceptron  | 0.76  | 0.79  |
| Decision Tree | 0.82  | 0.85  |

Podemos concluir que Extreme Gradient Boosting y Decision Tree Classifier son los modelos de aprendizaje más efectivos en este caso, y preservar los valores nulos resultó en una exactitud ligermante más alta, esto lo podemos considerar como algo bueno ya que reduce nuestra necesidad de hacer ajustes a nuestra base de datos.




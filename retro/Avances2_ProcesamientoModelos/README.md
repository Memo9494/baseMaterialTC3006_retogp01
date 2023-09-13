# Procesamiento de Modelos de Aprendizaje Máquina

## Introducción

En este documento, se presenta el proceso seguido en el proyecto, centrándose en la generación y comparación de modelos de aprendizaje máquina para abordar el reto planteado. En esta fase, se exploran diferentes tipos y configuraciones de modelos con el objetivo de seleccionar los más efectivos, cuyas decisiones se detallarán en la documentación. Para esto se hizo una investigación acerca de los modelos de clasificación, se encontraron varios modelos que cumplen con las características de nuestros datos para predecir de manera binaria un resultado, en nuestro caso la predicción de la muerte de un individuo en el Titanic. Entre estos modelos, se seleccionaron: KNeighborsClassifier, Support Vector Classifier (SVC), LogisticRegression, DecisionTreeClassifier, GaussianNB, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,  MLPClassifier.

Entre estos modelos, se tiene la intención de seleccionar sólo tres modelos, por lo que se realizó un proceso para seleccionar el modelo que será explicado en el documento, así mismo se documentó el proceso para la preparación de datos para cada modelo, así como la separación del dataset para entrenar y predecir. A continuación se describe el proceso de la selección e implementación de los modelos.


## Etapa 1: Investigación de modelos

La continuación del proyecto viene después de la limpieza de datos que se realizó en el entregable pasado, el siguiente paso fue investigar los modelos de predicción que nos interesan.

**K-Neighbors Classifier**: Utiliza el método de vecinos más cercanos para la clasificación. Las predicciones se basan en las clases de los ejemplos cercanos en el espacio de características. Requiere ajustar el número de vecinos y posiblemente otras métricas de distancia.

**SVC (Support Vector Classifier)**: Un clasificador que encuentra un hiperplano de separación óptimo entre clases. Puede usar diferentes funciones de kernel para transformar los datos en un espacio de mayor dimensión. Es útil para problemas de clasificación lineal y no lineal.

**Logistic Regression**: Realiza la clasificación utilizando la función logística para modelar la probabilidad de pertenencia a una clase. Es efectivo para problemas de clasificación binaria y multiclase cuando las clases son linealmente separables.

**Decision Tree Classifier**: Construye un árbol de decisión que divide recursivamente el espacio de características en regiones más puras. Es fácilmente interpretable, pero puede ser propenso al sobreajuste si no se controla adecuadamente.

**Gaussian NB**: Implementa el clasificador Bayesiano Ingenuo Gaussiano, asumiendo que las características son independientes y distribuidas normalmente. A pesar de su simplicidad, puede funcionar sorprendentemente bien en muchos casos.

**Random Forest Classifier**: Crea múltiples árboles de decisión y combina sus predicciones para mejorar la robustez y precisión del modelo. Puede manejar características categóricas y numéricas, y es menos propenso al sobreajuste que un solo árbol.

**Gradient Boosting Classifier**: Construye una secuencia de modelos que corrigen los errores del modelo anterior. Es útil para problemas de clasificación y regresión, y tiende a producir modelos de alta calidad.

**Decision Tree Classifier**: Construye un árbol de decisión que divide recursivamente el espacio de características en regiones más puras. Es fácilmente interpretable, pero puede ser propenso al sobreajuste si no se controla adecuadamente.

**Multi-layer Perceptron Classifier (MLP)**: Este clasificador implementa el algoritmo de aprendizaje profundo supervisado  MLP que se entrena mediante el método de retropropagación (backpropagation) está compuesto por múltiples capas de neuronas: capas de entrada, capas ocultas y capas de salida, cada una conectada a través de conexiones ponderadas (Neural network models (supervised), s. f.). Se escogió este modelo para realizar la predicción de los decesos en el titanic debido a su popularidad actual en la sociedad, puesto que este modelo es la base de las nuevas tecnologías que estamos usando como la nueva 

La librería cross_val_score se utiliza en la función que no hemos usado todavía en el proyecto para generalizar los análisis con gráficas y predicciones, así como se importa accuracy_score y metrics para obtener información de nuestros modelos y predicciones


## Etapa 2: Implementación de modelos

Para cada modelo se realizaron diferentes pruebas utilizando diferentes hiperparámetros y utilizando dos bases de datos diferentes: una en donde eliminamos los pasajeros que contienen datos nulos y otra en donde a través del algoritmo de K-vecinos predecimos los valores nulos para no eliminar dichos datos. Se siguió el siguiente procedimiento:

![](https://github.com/Memo9494/baseMaterialTC3006_retogp01/blob/main/retro/Avances2_ProcesamientoModelos/Diagrama_Modelos.png)

<p align="center">Fig 1. Metodología de pruebas</p>

### Verificar los mejores modelos

Se determinaron pues las variables dependientes e independientes de la base de datos así como los datos de entrenamiento y los de testeo.

Después se crea un array de modelos el cuál tiene la función de almacenar los modelos para utilizar un ciclo para preparar, entrenar y predecir cada uno de nuestros modelos que importamos, además se hace un análisis de precisión con la librería accuracy score.

A continuación presentamos los scores de los modelos:

*   **K-Neighbors Classifier** - KNN :  0.643357
*   **Support Vector Classifier** - SVC : 0.671329
*   **Logistic Regression** - LR :  0.755245
*   **Decision Tree Classifier** - DT : 0.713287
*   **Gaussian NB** - GNB  0.755245
*   **Random Forest Classifie** - RF  0.769231
*   **Gradient Boosting Classifier** -  GB  0.783217
*   **Multi-Layer Perceptron Classifier** -  MLP  0.762238

Se puede apreciar que los modelos con mejor precisión son: Gradient Boosting, Multi-Layer Perceptron y Decision Tree Classifier, por lo que serán uno de los modelos que se utilizaran para hacer el análisis y predicción. Sin embargo, se harán algunos cambios en las parámetros para mejorar los resultados de la predicción.

### Pruebas

Una vez determinados los mejores modelos, se realizaron tres pruebas por modelo de aprendizaje para cada base de datos.

A continuación se presentan los mejores resultados de precisión obtenidos en cada modelo:

|  | Eliminando los valores nulos  | Preservando los valores nulos |
| ------------- | ------------- | ------------- |
| Gradient Boosting  | 0.82 | 0.78 |
| Xtreme Gradient Boosting  | 0.84  | 0.80  |
| Multi-Layer Perceptron  | 0.76  | 0.79  |
| Decision Tree | 0.82  | 0.84  |

En específico, en Gradient Boosting tomamos los mejores hyperparámetros e hicímos un análisis de significancia de las variables como input, y nos entregó la siguiente gráfica, este nos dá a entender que no necesitamos la variable de enbarking, ya que no parece tener relación con el resultado de decesos.

Vemos que los resultados son diferentes de acuerdo con la base de datos utilizada, sin embargo no se ve un patrón preciso que determine que una base es mejor que otra, solamente que en la mayoría de los modelos la base de datos en la que se preservan los datos con valores nulos predichos la precisión de los modelos es mayor. Así mismo se puede notar que el clasificador de árbol de decisión es el que tiene mayor precisión.


## Conclusión 

A lo largo de esta entrega, se ha llevado acabo un proceso exhaustivo de selección, implementación y comparación de modelos de aprendizaje automático con el fin de lograr el objetivo de predecir la supervivencia de los pasajeros en el Titanic. En el transcurso de esta fase, se estudiaron una variedad de algoritmos de clasificación y se evaluaron en diferentes configuraciones para determinar cuales ofrecen el mejor rendimiento en materia de las métricas de evaluación. 

Entre el conjunto de los modelos analizados, destacaron dos enfoques particulares: el algoritmo Xtreme Gradient Boosting (XGBoost) y el uso de redes neuronales a través del Multi-Layer Perceptron Classifier (MLP). Ambos métodos demostraron resultados prometedores en términos de precisión, recall y F1-Score. El algoritmo XGBoost, que implementa un sistema de decisiones basado en árboles, se resaltó por su capacidad para manejar conjuntos de datos complejos y generar predicciones precisas. En contraste, la red neuronal MLP mostró una capacidad para aprender patrones en los datos y adaptarse a relaciones no lineales, lo que contribuyó a su alto rendimiento.

Es importante mencionar que, los mejores rendimientos encontrados en los modelos fueron extremadamente mejores cuando se trabajó con los datos que no contenían valores nulos. La eliminación de registros con datos faltantes permitió a los modelos centrarse en relaciones significativas en los datos disponibles y generar predicciones más precisas. Así mismo, se experimentó con diferentes valores de hiper-parámetros y técnicas de regularización. Las iteraciones constantes permitieron identificar combinaciones óptimas que maximizaron la precisión y la generalización de los modelos.


## Referencias

Neural network models (supervised). (s. f.). Scikit Learn. https://scikit-learn.org/stable/modules/neural_networks_supervised.html

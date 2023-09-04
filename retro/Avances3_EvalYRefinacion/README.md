# Evaluación y Refinamiento de modelo

## Introducción

En este reporte será detallado nuestro proceso para separar la base de datos en dos partes distintas, una para el entrenamiento de nuestro modelo y una para hacer las pruebas. También se justificarán las modificaciones hechas a la base de datos y las variables seleccionadas. Finalmente compararemos los resulatados obtenidos por los diferentes modelos y los cambios que hicimos para refinarlos y mejorar su rendimiento.

## Separación de los datos

Para dividir nuestra base de datos en dos se uso la función 'train_test_split()', proveniente de la librería 'sklearn'. Esta función nos permite hacer esta división facilmente y siguiendo los parámetros deseados. En nuestro caso decidimos usar un ratio de 80-20 en nuestro split, en otras palabras, el 80% de nuestros datos fueron utilizados para entrenar el modelo y los restantes para verificar los resultados de sus predicciones. También se utilizó el parámetro 'stratify' de la función para asegurarse que nuestros dos datasets tuvieran aproximadamente el mismo ratio de sobrevivientes y así tener resultados más representativos.

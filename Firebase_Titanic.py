import firebase_admin as fad
from firebase_admin import credentials, db
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cred = fad.credentials.Certificate("titanicequipo1-firebase-adminsdk-xhgda-c5aec8f227.json")

app = fad.initialize_app(cred, {'databaseURL': 'https://titanicequipo1-default-rtdb.firebaseio.com/'})

# Obtener una referencia a la base de datos
ref = db.reference('/Titanic_Equipo1')
# Leer datos
data = ref.get()

URL = "https://titanicequipo1-default-rtdb.firebaseio.com/Titanic_Equipo1/.json"
URL_clase = "https://titanicequipo1-default-rtdb.firebaseio.com/Titanic_Equipo1/clase_var.json"
URL_edad = "https://titanicequipo1-default-rtdb.firebaseio.com/Titanic_Equipo1/edad_var.json"
URL_emb = "https://titanicequipo1-default-rtdb.firebaseio.com/Titanic_Equipo1/emb_var.json"
URL_sex = "https://titanicequipo1-default-rtdb.firebaseio.com/Titanic_Equipo1/sex_var.json"

# Datos de entrenamiento
data_titanic = pd.read_csv("data_titanic.csv")
data_titanic = data_titanic.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
data_titanic['Sex'].replace(['male', 'female'],
                        [0, 1], inplace=True)
data_titanic['Embarked'].replace(['S', 'C', 'Q'],
                        [0, 1, 2], inplace=True)
data_titanic.apply (pd.to_numeric, errors='coerce')
df = data_titanic
df_NonNull = df.dropna()
feature_names = ['Age', 'Sex', 'Pclass', 'Embarked', 'Fare']
X_NonNull = df_NonNull[feature_names] # variables predictoras
y_NonNull = df_NonNull['Survived']    # variable de respuesta
#Se divide el dataset en entrenamiento y test, con un ratio de 80-20
X_NonNull_train, X_NonNull_test, y_NonNull_train, y_NonNull_test = train_test_split(X_NonNull, y_NonNull, test_size=0.2, random_state=42, stratify=y_NonNull)
regularized_tree = DecisionTreeClassifier(max_depth=15, min_samples_split=5)
regularized_tree.fit(X_NonNull_train, y_NonNull_train)

while True:
    try:
        #Request de las caracterisitcas del usuario
        clase_req = requests.get(URL_clase)
        clase_data = clase_req.json()
        clase_data = str(clase_data).strip('"')

        edad_req = requests.get(URL_edad)
        edad_data = edad_req.json()
        edad_data = str(edad_data).strip('"')

        emb_req = requests.get(URL_emb)
        emb_data = emb_req.json()
        emb_data = str(emb_data).strip('"')

        sex_req = requests.get(URL_sex)
        sex_data = sex_req.json()
        sex_data = str(sex_data).strip('"')

        #Manejo de datos para predicción
        # String a Int
        edad = int(edad_data)
        clase = int(clase_data)
        
        # Asignación de cuota (Fare)
        if clase == 1:
            fare = 84.1546875
        elif clase == 2:
            fare = 20.66218
        else:
            fare = 13.675550101832993
        #Variables dummy
        if sex_data == "Hombre":
            sex_data = 0
        else:
            sex_data = 1
        if emb_data == "Southampton":
            emb_data = 0
        elif emb_data == "Cherbourg":
            emb_data = 1
        else:
            emb_data = 2

        #Caracteristicas del pasajero
        pas = {'Age': [edad], 'Sex': [sex_data], 'Pclass': [clase], 'Embarked': [emb_data], 'Fare': [fare]}
        df_pas = pd.DataFrame(data = pas)

        # Predicción
        pred = regularized_tree.predict(df_pas)
        pred = pred[0]
        if pred == 1:
            res = "SI"
        else:
            res = "NO"

        #Publicacion de predicción
        pred_data = "{\"predic\":\"" + str(res) + "\"}"
        pred_data_pub = requests.patch(URL, data=pred_data)
        pred_data_pub.json()

    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")
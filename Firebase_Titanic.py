import firebase_admin as fad
from firebase_admin import credentials, db

cred = fad.credentials.Certificate("titanicequipo1-firebase-adminsdk-xhgda-c5aec8f227.json")

app = fad.initialize_app(cred, {'databaseURL': 'https://titanicequipo1-default-rtdb.firebaseio.com/'})

# Obtener una referencia a la base de datos
ref = db.reference('/Titanic_Equipo1')
# Leer datos
data = ref.get()
print(data)
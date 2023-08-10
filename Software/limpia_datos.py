import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tabulate import tabulate

# Cargar datos
data_titanic = pd.read_csv("data_titanic.csv")
table = tabulate(data_titanic.head(), headers='keys', tablefmt='fancy_grid')
print(table)

#Características de los datos
print(data_titanic.info()) #Información del dataset
print(data_titanic.shape) #Dimensiones del dataset
print(data_titanic.describe()) #Estadísticas del dataset
print(data_titanic.head()) #Histogramas de las variables
data_titanic = data_titanic.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']) #Eliminamos las variables que no nos interesan
data_titanic['Sex'].replace(['male','female'],[0,1],inplace=True)
data_titanic['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data_titanic.apply (pd.to_numeric, errors='coerce') #Convertimos los datos a numéricos 

df = data_titanic.dropna
categorias = ['Males', 'Females']
colores = ['red','blue']
n_males = data_titanic['Sex'].value_counts()[0]
n_females = data_titanic['Sex'].value_counts()[1]

def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))


x = np.linspace(0, 1, 100)
mu = 0.5
sigma = 0.1
y = gaussiana(x, mu, sigma)
plt.plot(x, y, color='black')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
#molestar a lupita es sencillo cuando se habla de edad
x = np.linspace(0, 1, 100)
def razon_edad(x):
    return 229/x

y = razon_edad(x)
plt.plot(x, y, color='black')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

datos = pd.read_csv('altura_peso.csv', sep=',')

x = datos['Altura'].values
y = datos['Peso'].values

np.random.seed(2)
modelo = Sequential()
modelo.add(Dense(1, input_dim=1, activation='linear')) 

sgd = SGD(learning_rate=0.0004)
modelo.compile(loss='mean_squared_error', optimizer=sgd)

modelo.summary()

num_epochs = 100
batch_size = len(x)  
historial = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

capas = modelo.layers[0]
w, b = capas.get_weights()
print(f'Parámetros: w = {w[0][0]:.1f}, b = {b[0]:.1f}')

plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')

y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x, y)
plt.plot(x, y_regr, 'r')
plt.title('Datos originales y regresión lineal')
plt.show()
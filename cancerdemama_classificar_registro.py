#CLASSIFICANDO UM REGISTRO
#utilizando parametros obtidos no tuning

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential() 

#criando camada de entrada:

classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim=30))
classificador.add(Dropout(0.2)) #zera 20% dos neuronios da camada de entrada
    
#criando uma nova camada oculta
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal')) 
classificador.add(Dropout(0.2))

#criando camada de saída -> um neuronio, função de ativação
classificador.add(Dense(units = 1, activation = 'sigmoid'))
#não altera pois é um problema binário
     
 #compilador -> descida do gradiente:
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)

#adição de um novo dado, potencialmente maligno
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
previsao = classificador.predict(novo)
previsao = (previsao>0.5)
#É maligno


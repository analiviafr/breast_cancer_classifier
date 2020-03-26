#SALVANDO UMA REDE NEURAL
#para adição facilitada de dados futuros

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


#salvando rede neural
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')

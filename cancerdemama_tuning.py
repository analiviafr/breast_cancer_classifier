#VERSÃO TUNING
#esse exemplo demora 8hrs

import pandas as pd
import keras
from keras.models import Sequential #modelo sequencial
from keras.layers import Dense, Dropout ##camadas densas - neuronios ligados as camadas ocultas
from keras.wrappers.scikit_learn import KerasClassifier #para validação cruzada
from sklearn.model_selection import GridSearchCV
#pesquisa em grade para detectar os melhores parâmetros

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

#função para criação da rede neural:
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential() 

    #criando camada de entrada:
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim=30))
    classificador.add(Dropout(0.2)) #zera 20% dos neuronios da camada de entrada
    
    #criando uma nova camada oculta
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer)) 
    classificador.add(Dropout(0.2))

    #criando camada de saída -> um neuronio, função de ativação
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    #não altera pois é um problema binário
    
    
    #compilador -> descida do gradiente:
    classificador.compile(optimizer = optimizer, loss = loos, metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size':[10,30],
              'epochs': [50, 100].
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}
#quanto maior os valores, mais tempo demora

#faz a busca:
grid_search = GridSearchCV(estimator = classificador, param_grid = parametros,
                           scoring = 'accuracy', cv=5)

grid_search = grid_search.fit(previsores,classe)

melhores_parametros = grid_search.best_params_
melhor precisao = grid_search.best_score_

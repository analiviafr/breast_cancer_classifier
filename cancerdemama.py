import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

#base de dados de treinamento - descobrir atributos
#base de dados de teste - base de acertos e erros
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)
#75% dos registros será usado para teste e 25% será usado para treino

import keras

from keras.models import Sequential #modelo sequencial
from keras.layers import Dense #camadas densas
#camadas densas - neuronios ligados as camadas ocultas

#criando uma nova rede:
classificador = Sequential() 

#criando camada de entrada:
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30)) #quantidade de neuronios na camada oculta
#quantidade de neuronios na camada oculta
#funcao de ativação
# (entrada + saida)/2 (30+1 pq é binario)/2
#input dim -> tamanho da camada de entrada -> só na primeira camada

#criando uma nova camada oculta
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform')) #quantidade de neuronios na camada oculta
#quantidade de neuronios na camada oculta


#criando camada de saída:
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#descida do gradiente:
otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
#lr = valor inicial
#decay = valor que vai decrementando
#clip = prende a descida mais proximo ao minimo global

#compilador -> descida do gradiente:
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
#otimizador adam, pode usar outros
#metricas - acurácia

#encontrar correlação entre classes e previsores:
#treinamento
classificador.fit(previsores_treinamento, classe_treinamento,
                 batch_size = 10, epochs = 100)

#visualização dos pesos
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0)) #quantidade de posições
#1-pessos fluindo da camada de entrada a oculta (30,16)
#2-unidade de baias
pesos1 = classificador.layers[1].get_weights()
#ligação da primeira camada oculta com a segunda (16,16)
pesos2 = classificador.layers[2].get_weights()
#ligação da ultima camada oculta com a camada de saída (16,1)


#para conseguir a real acuracia, deve-se fazer com os dados de teste:
previsoes = classificador.predict(previsores_teste)
#devolve dados em probabilidade

#passando para binário:
previsoes = (previsoes>0.5) #passa true ou false

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
#compara as saidas reais ao aprendido

matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)
#resultados:
#1- valor da função de erro
#2- precisao
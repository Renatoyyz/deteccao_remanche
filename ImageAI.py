import os
import shutil
from datetime import datetime
import warnings
from datetime import datetime

import numpy as np
import tensorflow

# from cliente_temp import Service

# print('Versão Tensorflow: ', tensorflow.__version__)
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import seaborn as sns  # Para gerar visualizações
from sklearn.model_selection import StratifiedKFold

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

sns.set()
warnings.filterwarnings('ignore')


class NeuralAI:
    def __init__(self, img_list = [], h_img=32, w_img=32, threshold=0.5):
        self.img_list = img_list
        self.img_dict = None

        self.gerador_treinamento = None
        self.gerador_teste = None
        self.base_treinamento = None
        self.base_teste = None
        self.precisao = None
        self.precisao_val = None

        self.threshold = threshold

        self.model = 0  # Variável que carrega o modelo
        self.history = 0  # Variável que armazena dados do treinamento
        self.labelencoder = LabelEncoder()  # Classe que transforma variáveis categorica em numéricas
        self.fix_gpu()

    def fix_gpu(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    def convolutional(self, num_classes = None):
        self.model = Sequential()

        self.model.add(Conv2D(64, kernel_size=(10, 10), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.4))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(10, 10), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.4))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=num_classes, activation='softmax'))  # Camada de saída densa que sempre vai ser de acordo com a quantidade de classes
        # self.model.add(Activation('softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    # num_batch_size = 32 = de 32 em 32 audios paraserem treinados
    def neural_training(self, num_classes = None, num_epochs=80, num_batch_size=32, arqui_train='', arqui_test=''):

        self.convolutional(num_classes=num_classes)
        self.gerador_treinamento = ImageDataGenerator(
                                                    rescale=1./255,
                                                    rotation_range=7,
                                                    horizontal_flip=True,
                                                    shear_range=0.2,
                                                    height_shift_range=0.07,
                                                    zoom_range=0.2
                                                    )
        self.gerador_teste = ImageDataGenerator(rescale=1./255)

        self.base_treinamento = self.gerador_treinamento.flow_from_directory(arqui_train,
                                                                             target_size=(64, 64),
                                                                             batch_size=num_batch_size,
                                                                             class_mode='categorical')
        self.base_teste = self.gerador_teste.flow_from_directory(arqui_test,
                                                                 target_size=(64, 64),
                                                                 batch_size=num_batch_size,
                                                                 class_mode='categorical')


        for chave in self.base_treinamento.class_indices.keys():
            self.img_list.append(chave)

        np.savetxt('label_encoder_img.txt', self.img_list, fmt='%s')
        # np.save('label_encoder_img', base_treinamento.class_indices)

        checkpointer = ModelCheckpoint(filepath='saved_models/image_classification.hdf5', verbose=1,
                                       save_best_only=True)
        start = datetime.now()
        # self.history = self.model.fit(x=self.X_train, y=self.Y_train , batch_size = num_batch_size, epochs = num_epochs, validation_data = (self.X_val, self.Y_val), callbacks = [checkpointer], verbose = 1)
        # self.model.fit( self.base_treinamento, steps_per_epoch=self.base_treinamento.samples / num_batch_size,
        #                epochs=num_epochs, validation_data=self.base_teste,
        #                validation_steps=self.base_teste.samples / num_batch_size, callbacks=[checkpointer],
        #                verbose=True)

        self.model.fit(self.base_treinamento, steps_per_epoch = int(self.base_treinamento.samples / num_batch_size),
                            epochs = num_epochs, validation_data = self.base_teste,
                            validation_steps = int(self.base_teste.samples / num_batch_size), callbacks=[checkpointer])

        duration = datetime.now() - start
        self.precisao = self.model.evaluate(self.base_treinamento)
        self.precisao_val = self.model.evaluate(self.base_teste)

        np.save('precisao', self.precisao)
        np.save('precisao_val', self.precisao_val)

        # with open('precisao.txt', 'w') as arquivo:
        #     arquivo.write(self.precisao)
        #     arquivo.close()
        # with open('precisao_val.txt', 'w') as arquivo:
        #     arquivo.write(self.precisao_val)
        #     arquivo.close()

        print('Duração do treinamento: ', duration)
        print(f'Precisão base treino: {self.precisao[1]}\nLoss base treino: {self.precisao[0]}\n')
        print(f'Precisão base teste: {self.precisao_val[1]}\nLoss base teste: {self.precisao_val[0]}\n')

    # num_batch_size = 32 = de 32 em 32 audios paraserem treinados
    def neural_training_kfold(self, dados_k_fold = r"dados/",num_classes = None, n_split = 10, num_epochs=80, num_batch_size=32, arqui_train=r'dataset_fold/train', arqui_test=r'dataset_fold/test'):

        src = dados_k_fold
        # Esse comando garante que se pegue somente os diretórios, pois, as vezes, pode-se ter arquivos ou arquivos ocultos como .DS_Store
        files_dados = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]
        tamanho_dados = []
        for l in files_dados:
            d = [f for f in os.listdir(src+l+'/') if os.path.isfile(os.path.join(src+l+'/', f)) and ('.jpg' or '.jpeg' or '.png') in os.path.join(src+l+'/', f) ]
            tamanho_dados.append(len(d))
        
        files_dados_np = np.array(files_dados)

        # n_split = em quanto será dividida a base de dados
        # shuffle = Habilita a pegar os dados aleatóriamente
        # randon_state = para determinar que a aletoriedade da divisão de dados seja sempre a mesma
        kfold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 5)

        list_data_treino = {}
        list_data_teste = {}

        for f in files_dados:   
            dir_loc = os.path.join(src, f)
            files_loc = [d for d in os.listdir(dir_loc) if os.path.isfile(os.path.join(dir_loc, d)) and ('.jpg' or '.jpeg' or '.png') in os.path.join(dir_loc, d)  ]
           
            files_loc_np = np.array(files_loc)
            list_data_treino.update( {f'{f}':[]} )
            list_data_teste.update( {f'{f}':[]} )
            for indice_treinamento, indice_teste in kfold.split(files_loc_np, np.zeros(shape = (files_loc_np.shape[0], 1))):
                list_data_treino[f'{f}'].append(indice_treinamento)
                list_data_teste[f'{f}'].append(indice_teste)

        cl = [*list_data_treino]

        tamanho_total = 0

        for j in tamanho_dados:
            tamanho_total = (tamanho_total + j)#*n_split
        tamanho_total = tamanho_total*n_split

        for cl in [*list_data_treino]:
            dir_loc = os.path.join(src, cl)
            files_loc = [d for d in os.listdir(dir_loc) if os.path.isfile(os.path.join(dir_loc, d)) and ('.jpg' or '.jpeg' or '.png') in os.path.join(dir_loc, d) ]
            # d = [f for f in os.listdir(src+l+'/') if os.path.isfile(os.path.join(src+l+'/', f)) and ('.jpg' or '.jpeg' or '.png') in os.path.join(src+l+'/', f) ]
            files_loc_np = np.array(files_loc)
            
            t = len(list_data_treino[f'{cl}'])
            for index in range(t):

                arquivo_train = 'dataset_fold/train/'+f'dataset{index}/'+cl
                if not os.path.isdir(arquivo_train):
                    os.makedirs(arquivo_train)

                arquivo_test = 'dataset_fold/test/'+f'dataset{index}/'+cl
                if not os.path.isdir(arquivo_test):
                    os.makedirs(arquivo_test)

                treino = list_data_treino[f'{cl}'][index]
                test = list_data_teste[f'{cl}'][index]

                for arq in files_loc_np[treino]:
                    shutil.copy(dir_loc+'/'+f'{arq}', arquivo_train)
                for arq in files_loc_np[test]:
                    shutil.copy(dir_loc+'/'+f'{arq}', arquivo_test)

        source_train_fold = arqui_train
        source_train_fold_ = [f for f in os.listdir(source_train_fold) if os.path.isdir(os.path.join(source_train_fold, f))]
        source_test_fold = arqui_test
        source_test_fold_ = [f for f in os.listdir(source_test_fold) if os.path.isdir(os.path.join(source_test_fold, f))]

        tx = []

        for index in range(len(source_train_fold_)):

            self.convolutional(num_classes=num_classes)
            self.gerador_treinamento = ImageDataGenerator(
                                                        rescale=1./255,
                                                        rotation_range=7,
                                                        horizontal_flip=True,
                                                        shear_range=0.2,
                                                        height_shift_range=0.07,
                                                        zoom_range=0.2
                                                        )
            self.gerador_teste = ImageDataGenerator(rescale=1./255)

            source_loc = source_train_fold+'/'+source_train_fold_[index]+'/'

            self.base_treinamento = self.gerador_treinamento.flow_from_directory(source_loc,
                                                                                target_size=(64, 64),
                                                                                batch_size=num_batch_size,
                                                                                class_mode='categorical')
                        
            source_loc = source_test_fold+'/'+source_test_fold_[index]+'/'
            self.base_teste = self.gerador_teste.flow_from_directory(source_loc,
                                                                    target_size=(64, 64),
                                                                    batch_size=num_batch_size,
                                                                    class_mode='categorical')


            for chave in self.base_treinamento.class_indices.keys():
                self.img_list.append(chave)

            np.savetxt(f'label_encoder_img_fold-{index}.txt', self.img_list, fmt='%s')
            self.img_list.clear()

            checkpointer = ModelCheckpoint(filepath=f'saved_models/image_classification_fold-{index}.hdf5', verbose=1,
                                        save_best_only=True)
            start = datetime.now()
            # self.history = self.model.fit(x=self.X_train, y=self.Y_train , batch_size = num_batch_size, epochs = num_epochs, validation_data = (self.X_val, self.Y_val), callbacks = [checkpointer], verbose = 1)
            # self.model.fit( self.base_treinamento, steps_per_epoch=self.base_treinamento.samples / num_batch_size,
            #                epochs=num_epochs, validation_data=self.base_teste,
            #                validation_steps=self.base_teste.samples / num_batch_size, callbacks=[checkpointer],
            #                verbose=True)

            # self.model.fit(self.base_treinamento, steps_per_epoch = 1,
            #                     epochs = num_epochs, validation_data = self.base_teste,
            #                     validation_steps = 1, callbacks=[checkpointer])
            self.model.fit(self.base_treinamento, steps_per_epoch = int(self.base_treinamento.samples / num_batch_size),
                                epochs = num_epochs, validation_data = self.base_teste,
                                validation_steps = int(self.base_teste.samples / num_batch_size), callbacks=[checkpointer])

            duration = datetime.now() - start
            self.precisao = self.model.evaluate(self.base_treinamento)
            self.precisao_val = self.model.evaluate(self.base_teste)

            np.save(f'precisao_fold-{index}', self.precisao)
            np.save(f'precisao_val-{index}', self.precisao_val)

            

            tx.append([f'Precisão base treino - {index}: {self.precisao[1]}',f'Loss base treino - {index}: {self.precisao[0]}'])
            tx.append([f'Precisão base teste - {index}: {self.precisao_val[1]}',f'Loss base teste - {index}: {self.precisao_val[0]}'])
            tx.append(["###########################################################################################\n"])

            

            np.savetxt( f'info_kfolds.txt', tx, fmt='%s')

            print('Duração do treinamento: ', duration)
            print(f'Precisão base treino: {self.precisao[1]}\nLoss base treino: {self.precisao[0]}\n')
            print(f'Precisão base teste: {self.precisao_val[1]}\nLoss base teste: {self.precisao_val[0]}\n')    

    def neural_load_model(self, directory = 'saved_models/', indice = 0 ):

        if os.path.isfile( directory+f'image_classification_fold-{indice}.hdf5'):
            self.model, self.img_list = self.load_model_by_name(directory+f'image_classification_fold-{indice}.hdf5')
            if os.path.isfile(f'label_encoder_img_fold-{indice}.txt'):
                # Carrega, se tiver, o array que contem as classificações
                self.img_list = np.loadtxt(f'label_encoder_img_fold-{indice}.txt', dtype=str)
                self.img_list = to_categorical(self.labelencoder.fit_transform(self.img_list))
                self.precisao = np.load(f'precisao_fold-{indice}.npy')
                self.precisao_val = np.load(f'precisao_val-{indice}.npy')
            else:
                print('Casses de classificação não encontrados.\nDiretório não encontrado.')
                return False
            return True
        else:
            print('Modelo não encontrado.\nDiretório não encontrado.')
            return False

    # Função para carregar o modelo
    def load_model_by_name(self, directory = ''):
        model = load_model(directory)
        img_dict = sorted(list(self.img_list))

        return model, img_dict

    def get_info(self, data, sample_rate):
        print('Canais: ', len(data.shape))
        print('Número total de amostras: ', data.shape[0])
        print('Taxa de amostragem: ', sample_rate)
        print('Duração: ', len(data) / sample_rate)

    def predict_image(self, arquivo_img, info=False, classe_=[]):
        imagem_teste = image.load_img(arquivo_img,
                                      target_size=(64, 64))
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste /= 255  # Normalização
        imagem_teste = np.expand_dims(imagem_teste, axis=0)  # Poe no formato do tensorflow (1, 64, 64, 3)

        start = datetime.now()
        self.prediction_global = self.model.predict(imagem_teste)
        val_max = np.max(self.prediction_global)

        if (val_max >= self.threshold):
            prediction = self.prediction_global.argmax(axis=1)
            prediction = prediction.astype(int).flatten()
            prediction = self.labelencoder.inverse_transform((prediction))

            duration = datetime.now() - start
            if prediction in classe_:
                previsao_bool = (self.prediction_global > self.threshold)
                print("\n###############################################\n")
                print(f'Classificação/resultado geral: {self.prediction_global}\nValor boleano: {previsao_bool}\n')
                print(f'Classificação: {prediction} Confiança: {val_max}')
                print('Duração da predição: ', duration)
                print(f'Precisão modelo (acurácia): {self.precisao_val[1]}\nErro do modelo (loss val): {self.precisao_val[0]}')
                print("\n###############################################\n")
            else:
                print(f'Classe {classe_} não encontrada')

        else:
            print('Nenhuma classe com confiança aceitável.')


        # base_treinamento.class_indices


if __name__ == '__main__':

    # img = ['adults', 'children']
    meta = NeuralAI( threshold=0.5)
    # meta.neural_training(num_epochs=5,num_classes=2, num_batch_size=4, arqui_train='archive/train', arqui_test='archive/test')
    # meta.neural_training_kfold( dados_k_fold='dados_/', num_classes=2, n_split=4, num_epochs=2, num_batch_size=16, arqui_train='dataset_fold/train', arqui_test='dataset_fold/test')

    # cliente = Service()

    # cliente.connect()

    meta.neural_load_model(indice=0)
    arqui = ''
    while arqui.upper() != 'Q':
        arqui = input("Favor por o path da imagem ou 'Q' para sair.\n")
        if arqui.upper() != 'Q':
            classe = input("Favor digitar a classe que se deseja encontrar.\n")
            classe = classe.split(',')
        if arqui.upper() != 'Q':
            print('==============================================\n\n')
            meta.predict_image(arqui, info=True, classe_=classe)
 
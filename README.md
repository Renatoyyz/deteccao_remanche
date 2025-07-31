# Neural AI Deteção de imagem e audio

## Pré-requisitos

```text
Python 3.8 ou superior
Pip 20.0 ou superior
virtualenv 20.4 ou superior
```

### Instalação

Para fazer a instalação do requirements é necessario primeiramente criar um ambiente virtual para podermos utilizar o app.

**Linux**
instalação do virtualenv:

```text
sudo apt-get install python3-venv
```

**Mac**
instalação do venv

```text
pip install virtualenv
```

criando o ambiente virtual:

```text
python -m venv nome_do_ambiente
```

para ativar o ambiente virtual :

```text
source nome_do_ambiente/bin/activate
```

Com o ambiente devidamente ativado instalamos os requerimentos com o seguinte código:

```text
pip install -r requirements.txt
```

Se for desejável pode se instalar o gerenciador de pacote de versionamento do python

```text
brew update
brew install pyenv
```

Para detalhes vá ao link:</br>
[Como gerenciar diversas versões do Python e ambientes virtuais](https://www.freecodecamp.org/portuguese/news/como-gerenciar-diversas-versoes-do-python-e-ambientes-virtuais/)

## Metadados

Cada arquivo de dados de imagem ou áudio é composto por:

* O nome tem o seguinte formato: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, onde:
  * [fsID] = é o ID do arquivo da imagem da qual este trecho (fatia) foi retirado.
  * [classID] = um identificador numérico da classe a ser detectada. Como exemplo temos classes de som ambiente de um arquivo de dados tirado do site [Freesound](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz) que vai de (0-9).
    * 0 = ar_condicionado
    * 1 = buzina_de_carro
    * 2 = crianca_brincando
    * 3 = latido_de_cachorro
    * 4 = perfuracao
    * 5 = motor_em_marcha_lenta
    * 6 = tiro_de_arma
    * 7 = britadeira
    * 8 = sirene
    * 9 = musica_de_rua
  * [occurrenceID] = Um identificador numérico para distinguir diferentes ocorrências do som dentro da gravação original;
  * [sliceID] = Um identificador numérico para distinguir diferentes fatias tiradas da mesma ocorrência;

Essa estrutura pode ser usada tanto para audio quanto para imagem.

## Extrutura dos arquivos

* raiz
  * **saved_model**: Para salvar os modelos treinados
  * **TesteDeAudio**: Audios de testes
  * **TesteDeImagem**: Imagens de testes
  * **TreinoDeAudio**: Audios para treino
  * **TreinoDeImagem**: Imagens para treino

### Criação da estrutura da rede neural

Os espectrogramas extraídos dos arquivos de áudio são como imagens 2D, então podemos usar técnicas de classificação de imagens neles, especificamente Redes Neurais Convolucionais (CNN)!

A arquitetura desta rede neural foi definida com base em alguns testes realizados para obter o resultado esperado. A estrutura pode ser ajustada livremente e comparada aos resultados desta estrutura.

* Parâmetros:
  * `Sequential`, é a classe para criar a rede neural, pois uma rede neural nada mais é que uma sequência de camadas (camada e entrada, camadas ocultas, camada de saída);  
  * `kernel_size`, o tamanho do kernel (matriz) de convolução;
  * `activation`, função de ativação;
  * `input_shape`, na primeira camada este é o tamanho dos dados de entrada
  * Camada `MaxPooling1D`, que vai fazer a extração das características principais;
  * Camada `Conv1d`, uma rede neural convolucional que realiza a convolução ao longo de apenas uma dimensão;
  * Camada `Flatten`, para transformar de matriz em vetor;
  * Camada `Dense`, quando um neurônio de uma camada está ligado a todas os outros neurônios das outras camadas;
  * `Dropout`, técnica de regularização para diminuir o overfitting: [Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
  * `padding='same'`, indica que adicionamos uma nova coluna composta por somente 0 (zeros) e utilizamos toda a imagem: [Padding](https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/)

  ![Convulutional](Img/convolutional.png)

## Treinando o modelo

* `num_epochs`, número de épocas de treinamento
* `num_batch_size`, isto indica que vamos enviar de 32 em 32 recursos de áudio (32, 64, 96, 128,...8732)

ModelCheckpoint, para salvar o modelo enquanto faz o treinamento

* `filepath`, caminho onde será salvo o modelo. Para isto temos uma pasta no Drive chamada *saved_models*
* `verbose`, mostrar mensagens enquanto a rede neural é treinada
* `save_best_only = True`, para salvar o modelo somente quando houver uma melhora no resultado

Variáveis para efetuar a contagem do tempo de treinamento:

* `start`, pegando o horário atual de início do treinamento;
* `duration`, ao final do treinamento, subtrair a hora atual com hora de início do treinamento.

* `model_history` para armazenar o histórico de treinamento:
* `model.fit` para fazer o ajuste do pesos ao longo do treinamento
  * `X_train`, `Y_train`, dados de treinamento
  * `batch_size = num_batch_size` que definimos acima
  * `epochs = num_epochs` que também definimos acima
  * `validation_data=(X_test, Y_test)`, dados de teste para monitorarmos como está o percentual de acerto da rede neural a cada época
  * `callbacks=[checkpointer]`, checkpointer definido anteriormente
  * `verbose = 1`, para mostrar as mensagens
  
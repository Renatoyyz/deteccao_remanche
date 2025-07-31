# Estrutura do DarkNet

1 - Fazer o clone do github

```text
git clone https://github.com/AlexeyAB/darknet
```

2 - Entrar no diretório "darknet" e compilar o projeto em "c"

```text
cd darknet
make
```

3 - Baixar os modelos pre-treinados

Modelo COCO: YOLO V4

```text
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

Ou o Modelo OpenImages: YOLO V3

```text
wget https://pjreddie.com/media/files/yolov3-openimages.weights
```

Ou o Modelo Tiny-YOLO: YOLO V4

```text
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

4 - Executat os comandos de detacção

No YOLO V4 com COCO:

```text
./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg -thresh 0.98  -ext_output
```

Onde:

* ./darknet: o arquivo que executa a estrutura darknet
* detect: comando de detecção
* cfg/yolov4.cfg: arquivo que contém as configurações da rede convolucional
* yolov4.weghts: modelo que contém os pesos da rede neural treinada
* data/person.jpg: arquivo de imagem que se deseja detectar
* -thresh n.nn: Valor de confiança que se deseja das classes detectadas
* -ext_output: Coordenadas do boundbox das classes encontradas

No YOLO V3 com Openimage:

```text
./darknet detector test cfg/openimages.data cfg/yolov3-openimages.cfg yolov3-openimages.weights data/person.jpg -thresh 0.60  -ext_output
```

Onde:

* ./darknet: o arquivo que executa a estrutura darknet
* detector: comando de detecção
* test cfg/openimages.data: Arquivo que contém o nome das classes para detecção
* cfg/yolov3-openimages.cfg: arquivo que contém as configurações da rede convolucional
* yolov3-openimages.weights: modelo que contém os pesos da rede neural treinada
* data/person.jpg: arquivo de imagem que se deseja detectar
* -thresh n.nn: Valor de confiança que se deseja das classes detectadas
* -ext_output: Coordenadas do boundbox das classes encontradas

No YOLO V4 co Tiny-YOLO:

```text
./darknet detector test cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights data/dog.jpg -thresh 0.60  -ext_output
```

Onde:

* ./darknet: o arquivo que executa a estrutura darknet
* detector: comando de detecção
* test cfg/openimages.data: Arquivo que contém o nome das classes para detecção
* cfg/yolov4-tiny.cfg: arquivo que contém as configurações da rede convolucional
* yolov4-tiny.weights: modelo que contém os pesos da rede neural treinada
* data/dog.jpg: arquivo de imagem que se deseja detectar
* -thresh n.nn: Valor de confiança que se deseja das classes detectadas
* -ext_output: Coordenadas do boundbox das classes encontradas

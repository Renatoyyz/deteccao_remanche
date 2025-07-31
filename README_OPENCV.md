# Estrutura com YOLO e OpenCV

Originalmente o Yolo tem uma estrutura para trabalhar com camadas de convolução do DarkNet então se faz necessário fazer ajustes para trabalhar com o OpenCV. Esses ajustes são feitos no método  [**def create_dnn(self)**](https://github.com/N2Bit/NeuralAI/blob/main/OpenCV.py#L21).

A execução que utiliza GPU - CUDA, utiliza versão do opencv que suporta array 2-D no retorno de "getUnconnectedOutLayers()", que são as camadas convolucionais DNN (Deep Neural Network).
Mas execuções que utilizam CPU, esse retorno da suporte a array de 1-D. Os duas possibilidades de código estão descritos [aqui](https://github.com/N2Bit/NeuralAI/blob/main/OpenCV.py#L25) e temos um [stack overflow](https://stackoverflow.com/questions/69834335/loading-yolo-invalid-index-to-scalar-variable) sobre um erro a respito disso.

As imagens submetidas para a rede neural precisa estar em formato tipo "blob"
Quando extraímos uma imagem com o opencv essa imagem fica no formato do numpy.ndarray, usando o "blobFromImage" da rede neural construida, é feito um préprocessamento na imagem de forma a executar uma técnica de subtração média e o redicionamento da imagem. A subtração média corrige problemas de luminosidade da imagem e o redicionamento define um tamanho padrão para ser submetido a rede neural. [Código](https://github.com/N2Bit/NeuralAI/blob/main/OpenCV.py#L54-L61)

## Non-max supression

Em uma imagem, no processo de detecção, é gerado vários bounding box. O No-max supression elimina as caixas compartilhadas e caixas com baixa probabilidade de ser uma classe a ser detectável.

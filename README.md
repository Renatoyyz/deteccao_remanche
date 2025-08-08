# Neural AI Detecção de Imagem e Áudio

## Pré-requisitos

```text
Python 3.8 ou superior
Pip 20.0 ou superior
virtualenv 20.4 ou superior
```

### Instalação

Para fazer a instalação do requirements é necessário primeiramente criar um ambiente virtual para podermos utilizar o app.

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

---

## Orientações e Processos Envolvidos

### Estrutura dos Dados e Metadados

- Os arquivos de imagem e áudio seguem o padrão `[fsID]-[classID]-[occurrenceID]-[sliceID].wav` ou `.png`.
- As classes devem estar equilibradas, idealmente com mais de 45 imagens por classe para melhor desempenho em validação cruzada (KFold).
- Estrutura de pastas recomendada:
  - **saved_model**: Modelos treinados
  - **TesteDeAudio** / **TesteDeImagem**: Dados de teste
  - **TreinoDeAudio** / **TreinoDeImagem**: Dados de treino

### Pré-processamento de Imagens

- **Normalização**: Todas as imagens são normalizadas com `rescale=1./255` no `ImageDataGenerator`.
- **Equalização de Histograma**: Melhora o contraste das imagens, especialmente útil para destacar regiões metálicas.
- **Gaussian Blur**: Suaviza a imagem e reduz ruídos antes da detecção de contornos.
- **Canny**: Destaca os contornos das regiões metálicas, facilitando a diferenciação entre metal e plástico.
- **Script de processamento em lote**: Para aplicar equalização, blur e Canny em todas as imagens de um diretório, utilize o script `processa_imagens.py`.

### Treinamento do Modelo

- **Validação Cruzada (KFold)**: Recomenda-se usar `n_split=6` para datasets com mais de 45 imagens por classe.
- **Arquitetura da Rede Neural**:
  - Camadas convolucionais profundas com BatchNormalization, MaxPooling, Dropout e Dense.
  - Função de ativação ReLU nas camadas ocultas e softmax na saída.
  - Regularização com Dropout e L2 pode ser aplicada para evitar overfitting.
- **Parâmetros de Treinamento**:
  - `num_epochs=100` para permitir aprendizado suficiente.
  - `EarlyStopping` com `patience=15` para evitar parada precoce.
  - `num_batch_size=4` (pode ser ajustado conforme memória disponível).
- **Monitoramento**:
  - Salve os logs de precisão e loss por fold em arquivos como `info_kfolds.txt`.
  - Escolha o melhor modelo pelo maior valor de precisão de teste e menor loss de teste.

### Avaliação dos Modelos

- **Precisão**: Mede o percentual de acertos do modelo.
- **Loss**: Mede o erro do modelo; quanto menor, melhor.
- **Seleção do Melhor Modelo**: Escolha o modelo do fold com maior precisão de teste e menor loss de teste.

### Pós-processamento e Predição

- **Predição**: O sistema utiliza as áreas marcadas em JSON para recortar e processar as imagens antes da predição.
- **Interface**: O resultado da predição mostra apenas os pontos que não passaram, facilitando a inspeção.
- **Configuração**: As configurações de arquivos e câmera podem ser salvas e carregadas automaticamente via JSON.

### Recomendações Gerais

- Sempre aplique o mesmo pré-processamento nas imagens de treino e teste.
- Mantenha o dataset equilibrado entre as classes.
- Teste diferentes arquiteturas e parâmetros para encontrar o melhor desempenho.
- Documente os resultados dos folds para facilitar futuras análises e compartilhamento com a equipe.

---

## Exemplos de Scripts Úteis

### Processamento em lote de imagens

```python
import os
import cv2
import numpy as np

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao ler {img_path}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                blur = cv2.GaussianBlur(eq, (3, 3), 0)
                edges = cv2.Canny(blur, 120, 250)
                img_out = img.copy()
                img_out[edges > 0] = [0, 0, 255]  # destaca contornos em vermelho
                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, img_out)
                print(f"Processado: {out_path}")

if __name__ == "__main__":
    input_dir = input("Digite o diretório de entrada das imagens: ")
    output_dir = input("Digite o diretório de saída para imagens processadas: ")
    process_images(input_dir, output_dir)
```

---

## Referências

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Padding em CNNs](https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/)
- [Freesound Dataset](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)
- [Como gerenciar diversas versões do Python e ambientes virtuais](https://www.freecodecamp.org/portuguese/news/como-gerenciar-diversas-versoes-do-python-e-ambientes-virtuais/)

---

## Observações Finais

Este documento reúne as principais práticas e decisões tomadas durante o desenvolvimento e testes do sistema de detecção de imagem e áudio. Use-o como referência para futuras consultas ou para orientar novos membros da equipe sobre o funcionamento e os processos do projeto.
# Object Detector API

## Descrição da API

Este projeto contempla o código para a API REST utilizada para detectar os objetos defeituosos manufaturados aditivamente.

A API foi desenvolvida utilizando a linguagem de programação Python em conjunto com a biblioteca do TensorFlow Object Detection API. A ideia é receber uma imagem e então realizar a detecção e retornar o(s) retângulo(s) que representam as caixas delimitadoras dos objetos, bem como as demais informações do mesmo. As informações que podem ser obtidas são:

### Retorno do processamento

- Localização do objeto - posição de cada objeto na imagem, descrita na forma de retângulos (caixas delimitadoras);
- Classe - categoria à qual pertence o objeto;
- Confiança – nível de confiança do modelo para determinar se o objeto pertence à classe à qual está classificado;
- Defeituoso – determina se o objeto está defeituoso ou não;
​- Nível de defeito - ajuda os usuários a determinar se o objeto pode ser usado ou não.

A Figura a seguir apresenta um exemplo de saída do modelo
![Screenshot_1117](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/1dae5356-6458-4264-b4d8-0595b55f4dde)

### Endpoints

Os endpoints da API são:

- /index: Apenas para teste de comunicação com a API;
- /enviar-imagem: Realiza as detecções na imagem e retorna as informações na forma do objeto DetectedObject.

## Dependências

Para utilizar a API devem se instaladas algumas dependências, dentre elas:

```
pip install tensorflow==2.8.0 # Evitar erros
pip install pycocotools
pip install tf_slim
pip install tensorflow.io
pip install tensorflow-addons
pip install lvis
pip install tf-nightly
pip install tf-models-official
apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```

É importante instalar também as demais dependencias da API como o flask etc. Caso encontre alguma dependência não listada favor solicitar a inserção da mesma.

Obs.: Para executar o processo é necessário alterar o IP em que a API está executando. Para isso, obtenho o endereço IP do host, acesse a classe: Utils/Contants/ConstantsAPI e altere o endereço IP em API_HOST para o endereço IP do host. Inicie a API

# Funcionamento do processo

Primeiramente a imagem é recebida pela API no endpoint enviar-imagem depois o DetectorService é responsável realizar a detecção. A inicialização das classes referentes as redes neurais são iniciadas na inicialização da API para evitar lentidão durante a realização das detecções. 

- Primeiro é realizada a detecção do(s) objeto(s) presentes na imagem pela rede FASTER-RCNN;
- A saída é caixa delimitadora que define o objeto dentro da imagem;
- Partindo dessa informação o objeto é recortado e redimensionado;
- Então, para cada objeto encontrado na imagem ele é recortado e redimensionado sendo entrada para as demais redes;
- Para definir se o objeto é defeituoso é empregada uma CNN;
- Para definir o nível de defeito do objeto a MobileNet.
- Ao final tem-se os retângulos que definem a caixa delimitadora e as informações de cada objeto;
- O processo permite adição de novos modelos, bastando que esse modelo seja treinado e validado;
- Após ele pode ser adicionado;

A Figura a seguir apresenta o digrama de blocos do processo desenvolvido.

![Screenshot_2](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/be1f0deb-1b0c-4537-9225-78a6fc5775a4)

# Dataset

O dataset utilizado para treinar os modelos se encontra no link: https://github.com/LuckasMacedo2/manufactured-objects-defectives-dataset/tree/master. Mais informações podem ser encontradas no próprio repositório.

# Aplicação mobile

Afim de testar a API foi desenvolvida uma aplicação mobile que se encontra no repositório, mais informações podem ser encontradas no próprio repositório. Repositório: https://github.com/LuckasMacedo2/object-detection-app

# Referências:

SILVA, L. M. D.; ALCALÁ, S. G. S.; BARBOSA, T. M. G. D. A. PROPOSTA DE MODELOS DE INTELIGÊNCIA ARTIFICIAL PARA DETECÇÃO DE DEFEITOS EM PEÇAS MANUFATURADAS ADITIVAMENTE. 2022. 

SILVA, L. M. DA; ALCALÁ, S. G. S.; BARBOSA, T. M. G. DE A. Detecção de produtos manufaturados defeituosos utilizando modelos de inteligência artificial. 2022b. 

MACEDO, L.; GOMES, S. UMA REVISÃO SISTEMÁTICA SOBRE A DETECÇÃO DE OBJETOS DEFEITUOSOS PRODUZIDOS POR MANUFATURA ADITIVA. Anais ... Encontro Nacional de Engenharia de Produção, 27 out. 2023. 

SILVA, L. M. D. et al. ALGORITMOS DE APRENDIZAGEM PROFUNDA PARA DETECÇÃO DE OBJETOS DEFEITUOSOS PRODUZIDOS POR MANUFATURA ADITIVA. 2023.

# English

# Object Detector API

## API Description

This project includes the code for the REST API used to detect defective additively manufactured objects.

The API was developed using the Python programming language in conjunction with the TensorFlow Object Detection API library. The idea is to receive an image and then perform detection and return the rectangle(s) that represent the bounding boxes of the objects, as well as other information about them. The information that can be obtained is:

### Return from processing

- Object location - position of each object in the image, described in the form of rectangles (bounding boxes);
- Class - category to which the object belongs;
- Confidence – confidence level of the model to determine whether the object belongs to the class to which it is classified;
- Defective – determines whether the object is defective or not;
​- Defect level - helps users determine whether the object can be used or not.

The following figure presents an example of model output
![Screenshot_5](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/fc45eba8-ca75-43dc-ae5e-6e1a187b7946)


### Endpoints

The API endpoints are:

- /index: Only for testing communication with the API;
- /send-image: Performs detections on the image and returns the information in the form of the DetectedObject object.

## Dependencies

To use the API, some dependencies must be installed, including:

```
pip install tensorflow==2.8.0 # Avoid errors
pip install pycocotools
pip install tf_slim
pip install tensorflow.io
pip install tensorflow-addons
pip install lvis
pip install tf-nightly
pip install tf-models-official
apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```

It is important to also install other API dependencies such as flask, etc. If you find any dependency not listed, please request its insertion.

Note: To execute the process it is necessary to change the IP on which the API is running. To do this, I get the host's IP address, access the class: Utils/Contants/ConstantsAPI and change the IP address in API_HOST to the host's IP address. Launch the API

# Process operation

Firstly, the image is received by the API on the send-image endpoint, then the DetectorService is responsible for performing the detection. The initialization of classes relating to neural networks is initiated at API initialization to avoid slowdowns during detections.

- First, the object(s) present in the image are detected by the FASTER-RCNN network;
- The output is bounding box that defines the object within the image;
- Based on this information, the object is cropped and resized;
- Then, for each object found in the image, it is cropped and resized and is input to the other networks;
- To define whether the object is defective, a CNN is used;
- To define the defect level of the MobileNet object.
- At the end there are rectangles that define the bounding box and the information for each object;
- The process allows the addition of new models, as long as this model is trained and validated;
- After it can be added;

The following figure shows the block diagram of the developed process.

![Screenshot_4](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/745a3304-f2aa-4644-9194-784cb9ddfb6d)


# Dataset

The dataset used to train the models can be found at the link: https://github.com/LuckasMacedo2/manufactured-objects-defectives-dataset/tree/master. More information can be found in the repository itself.

# Mobile application

In order to test the API, a mobile application was developed and found in the repository. More information can be found in the repository itself. Repository: https://github.com/LuckasMacedo2/object-detection-app

# References:

SILVA, L. M. D.; ALCALÁ, S. G. S.; BARBOSA, T. M. G. D. A. PROPOSTA DE MODELOS DE INTELIGÊNCIA ARTIFICIAL PARA DETECÇÃO DE DEFEITOS EM PEÇAS MANUFATURADAS ADITIVAMENTE. 2022.

SILVA, L. M. DA; ALCALÁ, S. G. S.; BARBOSA, T. M. G. DE A. Detecção de produtos manufaturados defeituosos utilizando modelos de inteligência artificial. 2022b.

MACEDO, L.; GOMES, S. UMA REVISÃO SISTEMÁTICA SOBRE A DETECÇÃO DE OBJETOS DEFEITUOSOS PRODUZIDOS POR MANUFATURA ADITIVA. Anais ... Encontro Nacional de Engenharia de Produção, 27 out. 2023.

SILVA, L. M. D. et al. ALGORITMOS DE APRENDIZAGEM PROFUNDA PARA DETECÇÃO DE OBJETOS DEFEITUOSOS PRODUZIDOS POR MANUFATURA ADITIVA. 2023.

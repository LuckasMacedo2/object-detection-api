import sys
import os
import random

import datetime

import jsonpickle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

current_directory = os.path.dirname(os.path.abspath(__file__))
modelos_directory = os.path.join(current_directory, "..", "Modelos")
sys.path.append(modelos_directory)

imgs_directory = os.path.join(current_directory, "..", "img_saidas")
sys.path.append(imgs_directory)

class_directory = os.path.join(current_directory, "..", "Classes")
sys.path.append(class_directory)

utils_directory = os.path.join(current_directory, "..", "Utils")
sys.path.append(utils_directory)

from Classes.ModeloObjectDetectionAPI import ModeloObjectDetectionAPI
from Classes.DetectedObject import DetectedObject
from Utils.Constants import ConstantsPath, ConstantsDeteccao
from Processamento import Processamento

class DetectorService():
    _instancia = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(DetectorService, cls).__new__(cls)
            cls._instancia.iniciar_servico()

        return cls._instancia
    
    def iniciar_servico(self):
        self.constPath = ConstantsPath()

        self.carregar_faster()
        self.carregar_cnn()
        self.carregar_mobilenet()

        self.EXTENSAO_IMAGEM = '.jpg'

    def carregar_faster(self):
        self.faster_rcnn = ModeloObjectDetectionAPI('', self.constPath.LABEL_MAP_PATH, 0.9)
        self.faster_rcnn.carregar_modelo_disco(self.constPath.FASTER_PESOS)

    def carregar_cnn(self):
        with open(self.constPath.CNN, 'r') as json_file:
            json_saved_model = json_file.read() 
        self.cnn = tf.keras.models.model_from_json(json_saved_model)

        self.cnn.load_weights(self.constPath.CNN_PESOS)
        self.cnn.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])

    def carregar_mobilenet(self):
        self.mobilenet = load_model(self.constPath.MOBILENET_PESOS, compile=False)
        self.mobilenet.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def detectar_desenhar_bbox(self, imagem):
        imagem_np = imagem_np = np.array(imagem)
        output_dict = self.faster_rcnn.executar_deteccao_imagem(imagem_np)
        retorno_processado = Processamento.retornar_coordenadas_bbox(output_dict, imagem)

        imagem_saida = imagem.copy()
        for i, retorno in enumerate(retorno_processado):
            ymin = retorno[0][2]
            ymax = retorno[0][3]
            xmin = retorno[0][0]
            xmax = retorno[0][1]

            imgCrop = cv2.resize(imagem[ymin:ymax, xmin:xmax], ConstantsDeteccao.TAM_IMAGEM)
            imgTeste = np.expand_dims(imgCrop, axis=0) / ConstantsDeteccao.LIM_PIXELS

            previsaoLevel = np.argmax(self.mobilenet.predict(imgTeste, verbose=0))
            previsaoDefect = self.cnn.predict(imgTeste, verbose=0)

            defeituoso = ConstantsDeteccao.DEFEITUOSO if (previsaoDefect > ConstantsDeteccao.CNN_LIMIAR) * 1 else ConstantsDeteccao.NAO_DEFEITUOSO
            
            imagem_saida = self.desenhar_bboxes_labels(imagem_saida, retorno, previsaoLevel, defeituoso, ymin, ymax, xmin, xmax)

        return imagem_saida

    def detectar(self, imagem):
        imagem_np = imagem_np = np.array(imagem)
        output_dict = self.faster_rcnn.executar_deteccao_imagem(imagem_np)
        retorno_processado = Processamento.retornar_coordenadas_bbox(output_dict, imagem)
        altura, largura, canais = imagem.shape

        listaDeteccoes = [] 
        imagem_saida = imagem.copy()
        for i, retorno in enumerate(retorno_processado):
            ymin = retorno[0][2]
            ymax = retorno[0][3]
            xmin = retorno[0][0]
            xmax = retorno[0][1]

            imgCrop = cv2.resize(imagem[ymin:ymax, xmin:xmax], ConstantsDeteccao.TAM_IMAGEM)
            imgTeste = np.expand_dims(imgCrop, axis=0) / ConstantsDeteccao.LIM_PIXELS

            previsaoLevel = np.argmax(self.mobilenet.predict(imgTeste, verbose=0))
            previsaoDefect = self.cnn.predict(imgTeste, verbose=0)

            defeituoso = ConstantsDeteccao.DEFEITUOSO if (previsaoDefect > ConstantsDeteccao.CNN_LIMIAR) * 1 else ConstantsDeteccao.NAO_DEFEITUOSO

            listaDeteccoes.append(DetectedObject([ymin, ymax, xmin, xmax], defeituoso, previsaoLevel, retorno[1], retorno[2], altura, largura))
            
            imagem_saida = self.desenhar_bboxes_labels(imagem_saida, retorno, previsaoLevel, defeituoso, ymin, ymax, xmin, xmax)
        
        cv2.imwrite(f'img_{random.randint(1, 10000)}.jpg', imagem_saida)

        return listaDeteccoes

    def desenhar_bboxes_labels(self, imagem, retorno, previsaoLevel, defeituoso, ymin, ymax, xmin, xmax):

        classe = retorno[1]
        percentual = retorno[2]

        # Detector de objetos
        # BBOX do objeto
        cv2.rectangle(imagem, (xmin,ymin), (xmax,ymax), ConstantsDeteccao.LISTA_CORES_BBOX[classe - 1], 5)

        # Label
        text = f'{ConstantsDeteccao.DICT_CLASSES[classe]} - {(percentual * 100):.2f}%'
        cv2.rectangle(imagem, (xmin, ymin), (xmin + 155, ymin + 20), ConstantsDeteccao.LISTA_CORES_BBOX[classe - 1], -1)
        cv2.putText(imagem, text, (xmin + 5, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Detector de defeitos e nível de defeito
        text = f"{defeituoso} - Level {previsaoLevel}"
        cv2.rectangle(imagem, (xmin, ymax), (xmin + 195, ymax - 20), ConstantsDeteccao.LISTA_CORES_BBOX[classe - 1],  -1)
        cv2.putText(imagem, text, (xmin + 5, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return imagem



    def realizar_deteccao(self, img):      
        #imagem = self.detectar(imagem)
        #_, img_encoded = cv2.imencode(self.EXTENSAO_IMAGEM, imagem)

        #return img_encoded.tobytes()
        return self.detectar(img)

    def salvar_imagem(self, imagem):
        cv2.imwrite(f'{datetime.datetime.now()}{self.EXTENSAO_IMAGEM}', imagem)

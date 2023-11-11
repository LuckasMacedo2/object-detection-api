import os
import cv2
import numpy as np

import time


class YOLO():
    arquivo_cfg = ''
    arquivo_pesos = ''
    imagem = ''


    imagem_temp = 'temp/imgTemp.jpg'
    IMAGEM_SAIDA = 'predictions.jpg'
    
    def __init__(self, arquivo_cfg = '', arquivo_pesos = '', imagem = None):
        self.arquivo_cfg = arquivo_cfg
        self.arquivo_pesos = arquivo_pesos
        self.imagem = imagem

    def set_imagem(self, imagem):
        self.imagem = imagem
    
    def detectar(self):
        os.chdir('Modelos/YOLO')

        print('----------------------------------------')
        print(os.getcwd())
        print('----------------------------------------')

        
        cv2.imwrite(self.imagem_temp, self.imagem)

        self.imagem_temp =  '../' + self.imagem_temp

        os.chdir('darknet')
        os.system(f'./darknet detect cfg/{self.arquivo_cfg} {self.arquivo_pesos} {self.imagem_temp}')
        print('Esperando ...')
        return cv2.imread(self.IMAGEM_SAIDA)

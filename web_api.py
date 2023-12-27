from flask import Flask, request, Response, send_file, jsonify
from Utils.Constants import ConstantsAPI

from Services.DetectorService import DetectorService
from Classes.DetectedObjectEncoder import DetectedObjectEncoder
from Classes.DetectedObject import DetectedObject

import json
from http import HTTPStatus
import datetime

import cv2
import numpy as np

import datetime
import random

#mask = None
app = Flask(__name__)
detector = DetectorService()

@app.route('/index', methods=['GET'])
def home():
    return "Teste"

@app.route('/enviar-imagem', methods=['POST'])
def enviar_imagem():

    print('-------------------------------------------------------------------------------')
    print(f'{datetime.datetime.now()} >> Imagem recebida iniciando o processamento')
    imagem_binaria = request.files['image'].read()
    nparr = np.fromstring(imagem_binaria, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(f'img_{random.randint(1, 10000)}.jpg', imagem)
    listaDeteccoes = detector.realizar_deteccao(imagem)

    print(f"Detectados {len(listaDeteccoes)} objeto(s)")
    for obj in listaDeteccoes:
        print(str(obj))

    detected_objects_json = json.dumps([obj.__dict__ for obj in listaDeteccoes], cls=DetectedObjectEncoder)
    print('==============================================================================')

    return detected_objects_json

if __name__ == '__main__':
   app.run(host = ConstantsAPI.API_HOST, port=ConstantsAPI.PORT, debug = True)
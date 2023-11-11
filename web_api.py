from flask import Flask, request, Response, send_file
from Utils.Constants import ConstantsAPI

from Services.DetectorService import DetectorService
from Classes.DetectedObjectEncoder import DetectedObjectEncoder

import json
from http import HTTPStatus
import datetime

import cv2
import numpy as np

#mask = None
app = Flask(__name__)
detector = DetectorService()

@app.route('/index', methods=['GET'])
def home():
    return "Teste"

@app.route('/enviar-imagem', methods=['POST'])
def enviar_imagem():
    imagem_binaria = request.files['image'].read()
    nparr = np.fromstring(imagem_binaria, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    listaDeteccoes = detector.realizar_deteccao(imagem)

    detected_objects_json = json.dumps([obj.__dict__ for obj in listaDeteccoes], cls=DetectedObjectEncoder)

    return detected_objects_json

if __name__ == '__main__':
   app.run(host = ConstantsAPI.API_HOST, port=ConstantsAPI.PORT, debug = True)
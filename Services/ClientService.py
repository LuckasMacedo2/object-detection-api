import requests
import numpy as np
import cv2
import base64
import json
from PIL import Image
from io import BytesIO


class ClientService():
    
    def __init__(self, url = ''):
        self.url = url

    def post_image(self, img_file = None):
        """ post image and return the response """
        if str(type(img_file)) == "<class 'str'>":
            img_file = cv2.imread(img_file, cv2.COLOR_BGR2RGB)

        imagem_pillow = Image.fromarray(cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        imagem_pillow.save(buffer, format="JPEG")
        arquivos = {"image": ("image.jpg", buffer.getvalue())}
        response = requests.post(self.url, files=arquivos)

        imagem_bytes = response.content
        imagem_np = np.frombuffer(imagem_bytes, np.uint8)

        return cv2.imdecode(imagem_np, cv2.IMREAD_ANYCOLOR)
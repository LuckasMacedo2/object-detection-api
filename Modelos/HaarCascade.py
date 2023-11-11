import cv2

class HaarCascade():
    face_detector = None

    def __init__(self):
        pass

    def carregar_pesos(self, caminho_pesos):
        self.face_detector = cv2.CascadeClassifier(caminho_pesos)
    
    def realizar_deteccao(self, imagem, minSize = (100,100)):
        '''img = Imagem de entrada em RGB
           minSize = tamanho mínimo dos retângulos'''

        image_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        detections = self.face_detector.detectMultiScale(image_gray, minSize=minSize)

        for (x, y, w, h) in detections:
            cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return imagem


import cv2
from Services.ClientService import ClientService
from Utils.Constants import ConstantsAPI

URL_BASE =  f'http://{ConstantsAPI.API_HOST}:{ConstantsAPI.PORT}'
POST_IMAGE = '/enviar-imagem'
URL_POST_IMAGE = URL_BASE + POST_IMAGE

video_capture = cv2.VideoCapture(0)

cliente =  ClientService(URL_POST_IMAGE)

while True:
    ret, frame = video_capture.read()

    frame = cliente.post_image(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
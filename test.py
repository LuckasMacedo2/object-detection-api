from Services.DetectorService import DetectorService
import cv2

detector = DetectorService()


imagem = detector.realizar_deteccao(cv2.imread("F:\\Estudos\\Mestrado\\Projeto\\Datasetv3\\NewDS\\Mesclado\\DatasetVNew\\UFG_AM_447_1670863723.jpg")) #2
#imagem = detector.realizar_deteccao(cv2.imread("F:\\Estudos\\Mestrado\\Projeto\\Datasetv3\\NewDS\\Mesclado\\DatasetVNew\\UFG_AM_191_1670863721.jpg")) #3

# Mostre a imagem na janela
cv2.imshow("Imagem", imagem)

# Aguarde até que uma tecla seja pressionada e, em seguida, feche a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
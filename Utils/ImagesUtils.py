from matplotlib import pyplot as plt
import cv2

class ImagesUtils():
    def mostrar_imagem(imagem):
        fig = plt.gcf() # Limpa as configurações do gráfico
        fig.set_size_inches(30, 15)
        plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.axis('off')
        plt.show()
import os

class ConstantsAPI():
    API_HOST = '192.168.100.4' 
    PORT = 5001
    

class ConstantsPath():
    def __init__(self):
        self.LABEL_MAP_PATH = self.obter_path("Assets", "label_map.pbtxt")

        self.FASTER_PESOS = self.obter_path("SavedModels", "FASTER\saved_model")
        self.CNN = self.obter_path("SavedModels", "CNN\CNN.json")
        self.CNN_PESOS = self.obter_path("SavedModels", "CNN\cnn_pesos.hdf5")
        self.MOBILENET_PESOS = self.obter_path("SavedModels", "MobileNet\MobileNet.h5")


    def obter_path(self, path, arquivo):
        current_directory = os.getcwd()
        return os.path.abspath(os.path.join(current_directory, ".", path, arquivo))

class ConstantsDeteccao():
    CNN_LIMIAR = 0.42
    DICT_CLASSES = {1: "Part 1", 2: "Part 2", 3: "Part 3"}

    LISTA_CORES_BBOX = [
      (0,255,255),  # 1 - Part 1  - cyan
      (0,250,154),  # 2 - Part 2  - MediumSpringGreen
      (255,215,0)     # 3 - Part 3  - Gold
    ]

    TAM_IMAGEM = (128, 128)
    LIM_PIXELS = 255

    DEFEITUOSO = "Defective"
    NAO_DEFEITUOSO = "Not defective"    





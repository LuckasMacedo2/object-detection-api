'''
  Classe base para criar um modelo baseado na API de detecção de objetos do TensorFlow
'''
import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
modelos_directory = os.path.join(current_directory, "..", "Modelos/FASTER/research")
sys.path.append(modelos_directory)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow as tf
import tensorflow_text
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import cv2

class ModeloObjectDetectionAPI():
  nome_modelo = ''
  caminho_labels = ''
  indices_categorias = {}
  modelo = None
  output_dict = {}
  nivel_confianca = 0.0

  def __init__(self, nome_modelo = '', caminho_labels = '', nivel_confianca = 0.5):
    self.nome_modelo = nome_modelo
    self.caminho_labels = caminho_labels
    self.indices_categorias = label_map_util.create_category_index_from_labelmap(caminho_labels, use_display_name=True)
    self.nivel_confianca = nivel_confianca
    if nome_modelo != '':
      self.carregar_model()


  def carregar_model(self):
    '''
      Realiza o download e cria o modelo
    '''
    base_url = 'http://download.tensorflow.org/models/object_detection/'  # Link para baixar o modelo
    model_file = self.nome_modelo + '.tar.gz'

    model_dir = tf.keras.utils.get_file(
      fname=self.nome_modelo,
      origin=base_url + model_file,
      untar=True)

    model_dir = pathlib.Path(model_dir)/'saved_model'

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    self.modelo = model

  def carregar_modelo_disco(self, model_path):
    io_device = '/job:localhost'
    load_options = tf.saved_model.LoadOptions(experimental_io_device=io_device)
    self.modelo = tf.saved_model.load(str(model_path), options=load_options)

  def executar_deteccao_imagem(self, imagem):
    '''
      Realiza a detecção dos objetos em um única imagem
    '''
    imagem = np.asarray(imagem) # Convertendo para numpy

    input_tensor = tf.convert_to_tensor(imagem) # Convertendo para o formato do tensor
    input_tensor = input_tensor[tf.newaxis, ...] # Adicionando uma nova dimensão no tensor que diz respeito ao batch size

    # Executando a inferência
    self.output_dict = self.modelo(input_tensor)

    # Convertendo para array numpy e removendo a dimensão extra
    num_detections = int(self.output_dict.pop('num_detections'))

    self.output_dict = {key:value[0, :num_detections].numpy() for key, value in self.output_dict.items()}
    self.output_dict['num_detections'] = num_detections

    # Convertendo as classes para int
    self.output_dict['detection_classes'] = self.output_dict['detection_classes'].astype(np.int64)

    # Manipulando os modelos e as máscaras

    if 'detection_masks' in self.output_dict:
      # Corrigindo o tamanho da imagem e adicionado a caixa delimitadora
      detection_reframed = utils_ops.reframe_box_masks_to_image_masks(self.output_dict['detection_masks'],
                                                                      self.output_dict['detection_masks'], self.output_dict['detection_boxes'],
                                                                      imagem.shape[0], imagem.shape[1])
      detection_reframed = tf.cast(detection_reframed > self.nivel_confianca, tf.uint8)
      self.output_dict['detection_masks_reframed'] = detection_reframed.numpy()

    return self.output_dict

  def detectar_mostrar_inferencia(self, imagem):
    '''
      Mostra a inferência da imagem
    '''
    # Preparando a imagem para ser dada como entrada para a rede
    if type(imagem) is str:
      imagem = np.array(Image.open(imagem))
    output_dict = self.executar_deteccao_imagem(imagem)

    # Exibindo o resultado
    return self.mostrar_inferencia(output_dict, imagem)

  def mostrar_inferencia(self, output_dict, imagem_np):
    vis_util.visualize_boxes_and_labels_on_image_array(
    imagem_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    self.indices_categorias,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=False,
    line_thickness=8)

    return imagem_np
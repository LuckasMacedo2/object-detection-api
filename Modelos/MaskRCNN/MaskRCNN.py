import os
import sys
from mrcnn import utils
import mrcnn.model as modellib 
import cv2
import random
import numpy as np
import time

sys.path.append(os.path.join(os.path.abspath('./Modelos\\MaskRCNN\\Mask_RCNN'), 'samples\\coco\\'))
import coco

class MaskRCNN():
    ROOT_DIR = ''
    MODEL_DIR = ''
    IMAGE_DIR = ''
    COCO_MODEL_PATH = ''
    config = None
    rede = None

    def __init__(self):
        self.ROOT_DIR = os.path.abspath('./Modelos\\MaskRCNN\\Mask_RCNN')
        sys.path.append(self.ROOT_DIR) 
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, 'logs')
        self.IMAGE_DIR = os.path.join(self.ROOT_DIR, 'images')
        
        self.dataset = Dataset()
        self.config = InferenceConfig()
  
        if not os.path.exists(self.dataset.DATASET_MODEL_PATH):
            utils.download_trained_weights(self.dataset.DATASET_MODEL_PATH)
    
        self.rede = modellib.MaskRCNN(mode='inference', model_dir = self.MODEL_DIR, config= self.config)
        self.rede.load_weights(self.dataset.DATASET_MODEL_PATH, by_name=True)
        self.rede.keras_model._make_predict_function()

        
        
    
    def realizar_deteccao(self, lista_imagens):
        print(len(lista_imagens), type(lista_imagens))
        if type(lista_imagens) != 'list':
            lista_imagens = [lista_imagens]
        

        print('-----------------------')
        resultados = self.rede.detect(lista_imagens, verbose = 1)
        print('-----------------------')

        lista_imagens_processadas = []
        print(len(resultados))
        for i ,r in enumerate(resultados):
            lista_imagens_processadas.append(self.display_instances(lista_imagens[i], r['rois'], r['masks'], r['class_ids'], self.dataset.nome_classes, r['scores']))

        return lista_imagens_processadas

    # Adaptado do framework da MaskRCNN
    ## Gera cores aleatórias
    def random_colors(self, N, seed, bright=True):
        import colorsys
        # Para poder gerar cores mais distintas visualmente vamos gerar os valores no espaço de cor HSV, depois convertemos para RGB. 
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.Random(seed).shuffle(colors)
        return colors

    ## Aplica mascara na imagem 
    def apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = np.where(
                    mask == 1,
                    image[:, :, c] *
                    (1 - alpha) + alpha * color[c] * 255,
                    image[:, :, c])
        return image

    ## Exibe as instancias (objetos segmentados)
    ## Foi necessário modificar essa função para que ela retorne a imagem (para podermos usar em nosso código e salvar os frames do vídeo) 
    def display_instances(self, image, boxes, masks, class_ids, class_names,
                        scores=None, title="", figsize=(16, 16), ax=None,
                        show_mask=True, show_bbox=True, colors=None, captions=None):
        from matplotlib import patches, lines
        from skimage.measure import find_contours
        from matplotlib.patches import Polygon
        
        # Numero de instancias
        N = boxes.shape[0]
        if not N:
            print("\n*** Nenhuma instancia a exibir *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Gera cores aleatórias
        colors = colors or self.random_colors(len(self.dataset.nome_classes), 55)

        height, width = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masked_image = image.astype(np.uint32).copy()

        for i in range(N):
            color = colors[class_ids[i]]

            # Caixa delimitadora / bounding box
            if not np.any(boxes[i]):
                # Pula essa instancias se não há caixas delimitadoras. Provavelmente foi perdida após crop na imagem 
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')

            # Rótulo / Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            masked_image = cv2.putText(np.float32(masked_image), caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Mascara
            mask = masks[:, :, i]
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)

            # Poligono da mascara
            # adiciona um espaçamento Para garantir polígonos adequados para máscaras que tocam as bordas da imagem.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)

        return masked_image.astype(np.uint8)


class Dataset():
    nome_classes = ['BG','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    ROOT_DIR = os.path.abspath('./Modelos\\MaskRCNN\\Mask_RCNN')
    DATASET_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
    sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
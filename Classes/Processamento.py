import numpy as np

class Processamento():
    @staticmethod
    def retornar_coordenadas_bbox(output_dict, img, min_score_thresh = 0.8):
        boxes = np.squeeze(output_dict['detection_boxes'])
        classes = np.squeeze(output_dict['detection_classes'])
        scores = np.squeeze(output_dict['detection_scores'])
        #set a min thresh score, say 0.8
        bboxes = boxes[scores > min_score_thresh]
        classes = classes[scores > min_score_thresh]
        scores = scores[scores > min_score_thresh]

        #get image size
        im_width, im_height = img.shape[1], img.shape[0]
        final_box = []
        for i, box in enumerate(bboxes):
            ymin, xmin, ymax, xmax = box
            final_box.append([[int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)], classes[i], scores[i]])

        return final_box
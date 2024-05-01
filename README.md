# Object Detector API

## API Description

This project includes the code for the REST API used to detect defective additively manufactured objects.

The API was developed using the Python programming language in conjunction with the TensorFlow Object Detection API library. The idea is to receive an image and then perform detection and return the rectangle(s) that represent the bounding boxes of the objects, as well as other information about them. The information that can be obtained is:

### Return from processing

- Object location - position of each object in the image, described in the form of rectangles (bounding boxes);
- Class - category to which the object belongs;
- Confidence – confidence level of the model to determine whether the object belongs to the class to which it is classified;
- Defective – determines whether the object is defective or not;
​- Defect level - helps users determine whether the object can be used or not.

The following figure presents an example of model output
![Screenshot_5](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/fc45eba8-ca75-43dc-ae5e-6e1a187b7946)


### Endpoints

The API endpoints are:

- /index: Only for testing communication with the API;
- /send-image: Performs detections on the image and returns the information in the form of the DetectedObject object.

## Dependencies

To use the API, some dependencies must be installed, including:

```
pip install tensorflow==2.8.0 # Avoid errors
pip install pycocotools
pip install tf_slim
pip install tensorflow.io
pip install tensorflow-addons
pip install lvis
pip install tf-nightly
pip install tf-models-official
apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```

It is important to also install other API dependencies such as flask, etc. If you find any dependency not listed, please request its insertion.

Note: To execute the process it is necessary to change the IP on which the API is running. To do this, I get the host's IP address, access the class: Utils/Contants/ConstantsAPI and change the IP address in API_HOST to the host's IP address. Launch the API

# Process operation

Firstly, the image is received by the API on the send-image endpoint, then the DetectorService is responsible for performing the detection. The initialization of classes relating to neural networks is initiated at API initialization to avoid slowdowns during detections.

- First, the object(s) present in the image are detected by the FASTER-RCNN network;
- The output is bounding box that defines the object within the image;
- Based on this information, the object is cropped and resized;
- Then, for each object found in the image, it is cropped and resized and is input to the other networks;
- To define whether the object is defective, a CNN is used;
- To define the defect level of the MobileNet object;
- At the end there are rectangles that define the bounding box and the information for each object;
- The process allows the addition of new models, as long as this model is trained and validated;
- After it can be added;

The following figure shows the block diagram of the developed process.

![Screenshot_4](https://github.com/LuckasMacedo2/object-detection-api/assets/33878052/745a3304-f2aa-4644-9194-784cb9ddfb6d)


# Dataset

The dataset used to train the models can be found at the link: https://github.com/LuckasMacedo2/manufactured-objects-defectives-dataset/tree/master. More information can be found in the repository itself.

# Mobile application

In order to test the API, a mobile application was developed and found in the repository. More information can be found in the repository itself. Repository: https://github.com/LuckasMacedo2/object-detection-app

# References:

SILVA, L. M.; ALCALÁ, S. G. S.; BARBOSA, T. M. G. A.; ARAÚJO, R. Object and defect detection in additive manufacturing using deep learning algorithms. Production Engineering, p. 1-14, 2024. DOI: http://dx.doi.org/10.1007/s11740-024-01278-y

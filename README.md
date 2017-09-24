# ObjectDetection
keras implementation of Faster R-CNN, for image object detection of woman apparels in ecommerce products. The original code is from https://github.com/yhenon/keras-frcnn. Some pre-processing and model modification is required for your own dataset.

Training Datasets:
1. DeepFashion dataset
    "Fashion Landmark detection" dataset is used and can be downloaded from https://www.dropbox.com/sh/d4kj5owfoq1iio6/AACuupJTpw9Waw_Ri2uk60twa?dl=0. Extracting the image file requires author's authentication, please contact the author for password.

2.  Artelab Dataset
  "Objects segmentation in the fashion field" dataset is used and can be downloaded from  http://artelab.dista.uninsubria.it/downloads/datasets/fashion_field/object_segmentation/object_segmentation.html.
  
USAGE:

Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.

requirements:
keras version: 2.0.2
tensorflow: 1.2.1
numpy: 1.13.1
cv2: 3.2.0


Training:
We trained on the DeepFashion and Artelab Datasets separately, since the annotation file for the two datasets are in different format and requires different pre-processing codes.
1. DeepFashion Dataset
Before training, please download the dataset and the annotation file "list_bbox_consumer2shop.txt", and place in the root directory.

The full command is:
python train_frcnn.py -o simple -p "list_bbox_consumer2shop.txt"

train_frcnn.py is used to train the DeepFashion model, with a pre-trained weights from ResNet50. -o shows the parser to preprocess the annotation file. -p indicates the image directory. The weights are saved in the format of "model_frcnn.hdf5_epoch_n" where n shows the epochs the weight is saved

2. Artelab Dataset
Before training, please download the dataset and the annotation file, and place in the directory of "artelab".

The full command is:
python train.py -o artelab -p artelab/ObjectsSegmentationFashion_v1.0/

train.py is used to train the Artelab model, with a pre-trained weights from ResNet50. -o shows the parser to preprocess the annotation file. -p indicates the image directory. The weights are saved in the format of "artelab.hdf5_epoch_n" where n shows the epochs the weight is saved

Prediction:
The two datasets cover different types of woman apparels, so the predictions are combined.

python test_combine.py -p "test"

-p indicates the directory where test images are placed.
The predicted images are labeled with bounding boxes and tags. They are saved in the "predicted_image" directory. 50 images from the dataset are selected for object detection.
Note: please replace the model_rpn.load_weights() and model_classifier.load_weights() with trained weights from the datasets.


Example output:
![alt text](https://github.com/sth4k/ObjectDetection/blob/master/predicted_image/1.png)
![alt text](https://github.com/sth4k/ObjectDetection/blob/master/predicted_image/2.png)
![alt text](https://github.com/sth4k/ObjectDetection/blob/master/predicted_image/20.png)



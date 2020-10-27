"""

Author: Pedro F. Proenza

TODO:
`- f16
 - Test visualization
 - video

This source modifies and extends the work done by:

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License
Written by Waleed Abdulla

------------------------------------------------------------

Usage:

    # First make sure you have split the dataset into train/val/test set. e.g. You should have annotations_0_train.json
    # Otherwise, You can do this by calling
    python3 split_dataset.py --dataset_dir ../data

    # Train a new model starting from pre-trained COCO weights on train set split #0
    python3 -W ignore detector.py train --model=coco --dataset=../data --class_map=./taco_config/map_3.csv --round 0

    # Continue training a model that you had trained earlier
    python3 -W ignore detector.py train  --dataset=../data --model=path/to/weights.h5 --class_map=./taco_config/map_3.csv --round 0

    # Continue training the last model you trained with image augmentation
    python3 detector.py train --dataset=../data --model=last --round 0 --class_map=./taco_config/map_3.csv --use_aug

    # Test model image by image
    python3 detector.py test --dataset=../data --model=last --round 0 --class_map=./taco_config/map_3.csv

    # Run COCO evaluation on the last model you trained
    python3 detector.py evaluate --dataset=../data --model=last --round 0 --class_map=./taco_config/map_3.csv

    # Check Tensorboard
    tensorboard --logdir ./models/logs


"""

import os
import time
import numpy as np
import json
import csv
import random
from imgaug import augmenters as iaa

import sys
sys.path.insert(0, "detector/")

from dataset import Taco
import model as modellib
from model import MaskRCNN
from config import Config
import visualize
import utils
import matplotlib.pyplot as plt

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import time

#load image
from keras.preprocessing.image import load_img, img_to_array

# Root directory of the models
ROOT_DIR = os.path.abspath("./detector/models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def test_dataset(model, dataset, image_arr):
    # image_test = load_img('can.jpg')
    # image_arr = img_to_array(image_test)

    # image_list = []

    # for i in range(0, 2):
    #     image_list.append(img_to_array(image_test))

    print("Start detection!!!!!!!!!!!")
    start_time = time.time()
    r = model.detect([image_arr], verbose=0)[0]
    # r = model.detect(image_list, verbose=0)[0]
    print("---{}s seconds---".format(time.time()-start_time))
    print("Finish detection!!!!!!!!!!!")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))

        # Display predictions
    visualize.display_instances(image_arr, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], title="Predictions", ax=ax1)

    print(r['rois'])
    tmp_name = 'test_temp.jpg'
    plt.savefig(tmp_name)
    print("Finished!!!")

def init():

    # off CUDA
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Mask R-CNN on TACO.')

    args = parser.parse_args()
    args.command = "test"
    args.model = "last"
    args.dataset = "./data"
    args.round = 0
    args.class_map = "./detector/taco_config/map_10.csv"

    # Read map of target classes
    class_map = {}
    with open(args.class_map) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}

    # Test dataset
    dataset_test = Taco()
    taco = dataset_test.load_taco(args.dataset, args.round, "test", class_map=class_map, return_taco=True)
    dataset_test.prepare()
    nr_classes = dataset_test.num_classes

    class TacoTestConfig(Config):
        NAME = "taco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.9
        NUM_CLASSES = nr_classes
    config = TacoTestConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Find last trained weights
    model_path = model.find_last()[1]

    # Load weights
    model.load_weights(model_path, by_name=True)

    augmentation_pipeline = None

    return model, dataset_test
if __name__ == '__main__':
    print("test")

    image_test = load_img('can.jpg')
    image_arr = img_to_array(image_test)

    model, dataset = init()

    test_dataset(model, dataset, image_arr)
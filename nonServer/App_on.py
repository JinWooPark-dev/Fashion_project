from io import BytesIO
from tkinter import Image
import numpy as np
from flask import Flask, render_template, request
import json
import base64
import os
import os
import time
import numpy as np
import json
import csv
import random
from imgaug import augmenters as iaa
import tensorflow as tf

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
import cv2

# load image
from keras.preprocessing.image import load_img, img_to_array

# Root directory of the models
ROOT_DIR = os.path.abspath("./detector/models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# file_dir = "./images/frame120.jpg"

app = Flask(__name__)

count = 0
model = 0
dataset = 0

#
# def process1():
#     start = time.perf_counter()
#     time.sleep(0.2)

# def moVideo(file):
#     cap = cv2.VideoCapture(file)
#     model, dataset = init()
#
#     count = 0
#     while(cap.isOpened()):
#         if count > 1:
#             ret, frame = cap.read()
#             count += 1
#             # continue
#             break
#
#         ret, frame = cap.read()
#         image_arr = img_to_array(frame)
#         # image_arr = np.array(image_arr)
#         a, b = test_dataset(model, dataset, image_arr, count)
#
#         # print(a,b)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         count += 1
#
#     cap.release()
#     # cv2.destroyAllWindows()
#     print(a)
#     print(b)
#     #
#     return a, b
# file_dir = os.path.basename('/data/home/ldy/practice/practice_flask/Trash_detection/images/frame120.jpg')

def test_dataset(model, dataset, image_arr, count):
    print("Start detection!!!!!!!!!!!")
    start_time = time.time()
    r = model.detect([image_arr], verbose=0)[0]
    # r = model.detect(image_list, verbose=0)[0]
    print("---{}s seconds---".format(time.time() - start_time))
    print("Finish detection!!!!!!!!!!!")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))

    trashClass = ["BG", "Plastic", "Can", "Other"]

    print(r['class_ids'])
    for i in range(0, len(r['class_ids'])):
        if r['class_ids'][i] == 1 or r['class_ids'][i] == 2 or r['class_ids'][i] == 5 or r['class_ids'][i] == 6 or \
                r['class_ids'][i] == 10:
            r['class_ids'][i] = 1
        elif r['class_ids'][i] == 3 or r['class_ids'][i] == 9:
            r['class_ids'][i] = 2
        elif r['class_ids'][i] == 4 or r['class_ids'][i] == 7 or r['class_ids'][i] == 8:
            r['class_ids'][i] = 3
        else:
            r['class_ids'][i] = 0

    print(r['class_ids'])
    # Display predictions
    # visualize.display_instances(image_arr, r['rois'], r['masks'], r['class_ids'],
    #                             dataset.class_names, r['scores'], title="Predictions", ax=ax1)
    visualize.display_instances(image_arr, r['rois'], r['masks'], r['class_ids'],
                                trashClass, r['scores'], title="Predictions", ax=ax1)

    print(r['rois'])
    print(dataset.class_names)
    tmp_name = './output/temp_' + str(count) + '.jpg'
    plt.savefig(tmp_name)
    a = r['rois']
    b = r['class_ids']
    # data = a.tolist() + b.tolist()

    print("Finished!!!")

    return a, b


def init():
    # off CUDA
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Mask R-CNN on TACO.')

    args = parser.parse_args()
    args.command = "test"
    args.model = "last"
    args.dataset = "./data"
    args.round = 0
    # args.class_map = "./detector/taco_config/map_10.csv"
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
        DETECTION_MIN_CONFIDENCE = 0.8
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


# def App_test():
#     img_dir = './images/'
#     while True:
#         # process1()
#         global pre_num
#         global next_num
#         pre_num = len(os.listdir(img_dir))
#         if pre_num == next_num:
#             break
#         for i in range(1,len(os.listdir(img_dir)) + 1):
#             dir = (f'./images/webcam/frame{i}.jpg')
#
#             with open(dir, 'rb') as f:
#                 encoded_image = base64.b64encode(f.read())
#                 encoded_image = encoded_image.decode('utf-8')
#
#     next_num = len(os.listdir(img_dir))



# @app.route('/image', methods=['POST'])
@app.route('/image', methods=['POST'])
def image_segmentation():
    global count
    if count == 0:
        print("count!!!!!!!!!!!!!!!")
        global model
        global dataset
        model, dataset = init()
        count += 1
    # model, dataset = init()
    # tf.keras.backend.clear_session()
    if request.method == 'POST':
        response = {}
        response['message1'] = 'Hello World1!'

        if 'data' in request.json:
            response['message2'] = 'Hello World2!'
            i = request.json['num']
            # response['aaaaaaaaaaaaaaaa'] = i

            # print(i)
            # file_dir = "./images/frame120.jpg"
            file_dir = (f'./images/frame{i}.jpg')
            # print(file_dir)
            # # file = './images/frame120.jpg'
            #
            tempImage = cv2.imread(file_dir, 1)
            # print("1")
            image_arr = img_to_array(tempImage)
            # print("2")
            a, b = test_dataset(model, dataset, image_arr, i)
            a = [[123, 123, 123, 123]]
            if len(a) > 0:
                response['aaaaaaaaaaaaaaaa'] = a[0][0]
            else:
                response['aaaaaaaaaaaaaaaa'] = "test"
            # print("3")
            print(type(a))
            # a = base64.b64encode(a).decode('utf-8')
            # print("4")
            print(type(b))
            # b = base64.b64encode(b).decode('utf-8')
            # print("5")
            #
            # response['a'] = a
            # response['b'] = b
            # print("6")
        return json.dumps(response, indent=4, ensure_ascii=True)

    #     response['mg2'] = 'return from server'  # 안나옴
    #
    # return json.dumps(response, indent=4, ensure_ascii=True)


if __name__ == '__main__':
    # global count
    # count = 0

    # model, dataset = init()
    app.run(host='0.0.0.0', port=5000, debug=True)



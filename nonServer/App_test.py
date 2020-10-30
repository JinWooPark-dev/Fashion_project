import requests
import json
import base64
import os
import time

url = "http://101.101.208.247:5000/image"
img_dir = './images/webcam'
next_num = 0

def process1():
    start = time.perf_counter()
    time.sleep(2)

while True:
    process1()
    pre_num = len(os.listdir(img_dir))
    if pre_num == next_num:
        break
    for i in range(1,len(os.listdir(img_dir)) + 1):
        dir = (f'./images/webcam/img{i}.jpg')

        with open(dir, 'rb') as f:
            encoded_image = base64.b64encode(f.read())
            encoded_image = encoded_image.decode('utf-8')


        payload = {}
        # img_dir = './encodingImage/output1.bin'
        payload["data"] = encoded_image
        payload["num"] = i

        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        print(response.text)

    next_num = len(os.listdir(img_dir))


# json 형태로 하나씩 받고 싶다면
# result = response.json() # dict
# print(result["message"])
# 혹은
# response.json()["message"])
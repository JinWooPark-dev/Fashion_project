from io import BytesIO

import requests
import json
import base64
import os
import time
import asyncio
import numpy as np

import array as ar

url = "http://101.101.208.247:5000/image"
# img_dir = './images/webcam'
# next_num = 0

def process1():
    start = time.perf_counter()
    time.sleep(5)

def App_test():
    # while True:
    #     process1()
    #     global pre_num
    #     global next_num
    #     pre_num = len(os.listdir(img_dir))
    #     if pre_num == next_num:
    #         break
    #     for i in range(1,len(os.listdir(img_dir)) + 1):
    #         dir = (f'./images/webcam/img{i}.jpg')
    #
    #         with open(dir, 'rb') as f:
    #             encoded_image = base64.b64encode(f.read())
    #             encoded_image = encoded_image.decode('utf-8')

    # for i in range(120, 130):
    payload = {}
    # img_dir = './encodingImage/output1.bin'
    payload["data"] = "name.mp4"

    for i in range(120, 131):
        payload["num"] = i
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        print(response.json())
        # aa = response.json()['x1']

        # if aa == 'test':
        #     continue
        # else:
        #     pass
            # print(type(aa))
            # a = base64.b64decode(aa)
            # print(type(a))
            # temp = json.loads(a)
            # print(temp)
            # print("test!!!!!!")
            # print(aa)
            # if len(aa) <= 0:
            #     print("test")
            # else:
            #     print(aa)

            # print(base64.b64encode(a).decode('utf-8'))
            # print(BytesIO(base64.b64decode(aa)).getvalue())
            # byte = BytesIO(base64.b64decode(aa))
            # print(np.frombuffer(byte, dtype=np.float64))
        # try:
        #     # response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        #     result = response.json()  # dict
        #     print(result)
        # except Exception as e:
        #     print(e)
        #     exit()
    #
    # print("before wait!!!!!!!")
    # process1()
    # print("after wait!!!!!!!")

    # try:
    #     response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    #     result = response.json()  # dict
    #     print(result)
    # except Exception as e:
    #     print(e)
    #     exit()
    # print(response.json()['seg'])

    # next_num = len(os.listdir(img_dir))


# json 형태로 하나씩 받고 싶다면
# result = response.json() # dict
# print(result["message"])
# 혹은
# response.json()["message"])

if __name__ == '__main__':
    App_test()
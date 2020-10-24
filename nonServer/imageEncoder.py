import base64
import os
import time
img_dir = './images/webcam'
# print(len(os.listdir(img_dir)))
next_num = 0

# 웹 캠이 1초에 1장을 찍기 때문에 2초간 대기하기 위한 함수
def process1():
    start = time.perf_counter()
    time.sleep(2)

while True:
    process1()
    pre_num = len(os.listdir(img_dir))
    if pre_num == next_num:
        break

    for i in range(1,len(os.listdir(img_dir))):
        with open(f'./images/webcam/img{i}.jpg', 'rb') as imagefile:
            byteform = base64.b64encode(imagefile.read())
        f = open(f'./encodingImage/output{i}.bin', 'wb')
        f.write(byteform)
        f.close()
    next_num = len(os.listdir(img_dir))
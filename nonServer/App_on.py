from io import BytesIO
from tkinter import Image
import numpy as np
from flask import Flask, render_template, request
import json
import base64
import os

app = Flask(__name__)

@app.route('/image', methods=['POST', 'GET'])
def print_hello():
    i = 0
    if request.method == 'POST':
        response = {}
        # response['message'] = 'Hello World!'
        if 'data' in request.json:
            response['message'] = 'Hello World!'
            data = request.json['data']
            num = request.json['num']
            # fh = open('decodedImage1.png', 'wb')
            # fh.write(base64.b64decode(data))
            # fh.close()
            print(data)
            try:
                # base64 decode
                # decoded_bytes = BytesIO(base64.b64decode(data))
                # img = Image.open(decoded_bytes).convert('RGB')
                # img = np.asarray(img)

                fh = open(f'./images/webcam/decodedImage{num}.png', 'wb')
                byte = data
                fh.write(base64.b64decode(byte))
                fh.close()
                response['message'] = 'Hello img!'
            except:
                response['message'] = 'could not open image'
                return json.dumps(response, indent=4, ensure_ascii=False)
        else:
            response['message'] = '"data" not found'
            return json.dumps(response, indent=4, ensure_ascii=False)
        # file = open('output1.bin', 'rb')
        # byte = file.read()
        # file.close()
        #
        # fh = open('decodedImage1.png', 'wb')
        # fh.write(base64.b64decode(byte))
        # fh.close()

        return json.dumps(response, indent=4, ensure_ascii=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
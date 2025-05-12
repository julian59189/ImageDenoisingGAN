from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import base64
from io import BytesIO
from test import *
import time
from PIL import Image
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/denoisify', methods=['GET', 'POST'])
def denoisify():
    if request.method == "POST":
        inputImg = request.files['file']
        outputImg = denoise(inputImg)
        # scipy.misc.imsave('static/output.png', outputImg)
        # If outputImg is a numpy array
        output_image = (outputImg * 255).astype(np.uint8)  # Denormalize if it's in [0, 1] range
        Image.fromarray(output_image).save('static/output.png')

        return jsonify(result="Success")


if __name__=="__main__":
    app.run(host="0.0.0.0",port="80")

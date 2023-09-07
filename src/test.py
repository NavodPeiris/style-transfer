import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import os.path as osp

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

from fastapi import FastAPI, UploadFile, File
import shutil
import os
from fastapi.responses import FileResponse
import socket

app = FastAPI()

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print("ip_address : ", ip_address)

@app.get('/styleTransfer/health')
async def hi():
    return {"response": "server running"}

@app.post('/styleTransfer/infer')
async def styleTransfer(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    # file1 is content image
    # file2 is style image

    source_folder = 'source_images/*'
    out_folder = 'result_images/*'

    # deleting result images
    for path in glob.glob(source_folder):
        if os.path.exists(path):
            os.remove(path)

    # deleting result images
    for path in glob.glob(out_folder):
        if os.path.exists(path):
            os.remove(path)

    save_directory = "./source_images"
    
    # Save the uploaded file
    file1_path = os.path.join(save_directory, file1.filename)
    file2_path = os.path.join(save_directory, file2.filename)

    with open(file1_path, "wb") as image:
        shutil.copyfileobj(file1.file, image)
        print("first image written")
    
    with open(file2_path, "wb") as image:
        shutil.copyfileobj(file2.file, image)
        print("second image written")

    base1 = osp.splitext(osp.basename(file1_path))[0]
    base2 = osp.splitext(osp.basename(file2_path))[0]
    print(base1 + base2)

    outpath = 'result_images/{:s}_rlt.png'.format(base1 + base2)

    content_image = load_image(file1_path)    # image that has non artistic content
    style_image = load_image(file2_path)      # image that is an art

    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    cv2.imwrite(outpath, cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

    return FileResponse(outpath)
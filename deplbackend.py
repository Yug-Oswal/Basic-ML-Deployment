from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np 
from PIL import Image

model = tf.keras.models.load_model(r'C:\Windows\System32\CatvDogClass')

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/about")
def disp_about():
    return render_template('about.html')

@app.get("/predict")
@app.route("/input")
def take_input():
    return render_template('input.html')

@app.post("/predict")
def give_inference():
    if 'image' not in request.files:
        return "No image uploaded"
    
    image = request.files['image']

    img = Image.open(image)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    target_size = (150, 150)

    img_resized = img.resize(target_size)

    x = np.array(img_resized)
    img_final = x / 255
    
    img_final = np.expand_dims(img_final, axis = 0)

    output = model.predict(img_final)

    if (output[0] > 0.5):
        return render_template('predict.html', prediction = "dog")
    else:
        return render_template('predict.html', prediction = "cat")



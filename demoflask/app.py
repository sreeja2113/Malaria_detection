from flask import Flask, request
from keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

model = load_model("malaria_cnn_model.h5")

def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    return np.reshape(image_array, (1, 64, 64, 3))

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image = Image.open(imagefile)
    image_array = preprocess_image(image)
    pred = model.predict(image_array)
    prediction = np.argmax(pred)

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)

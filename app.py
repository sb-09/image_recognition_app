import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Loading the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# preprocessing the image to fit model measurements
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# image recognition using the MobileNetV2 model
def recognize_image(image_path):
    try:
        preprocessed_img = preprocess_image(image_path)
        predictions = model.predict(preprocessed_img)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        recognized_object = decoded_predictions[0][1]
        return recognized_object
    except Exception as e:
        return str(e)
# route creation used to reach to the html file in templates folder
@app.route('/')
def index():
    return render_template('index.html')
#route creation to store file temporarily in uploads folder
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'result': 'No image provided.'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'result': 'No image selected.'}), 400

    if image:
        uploads_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

        os.makedirs(uploads_folder, exist_ok=True)

        image_path = os.path.join(uploads_folder, image.filename)
        image.save(image_path)

        recognized_object = recognize_image(image_path)

        os.remove(image_path)

        return jsonify({'result': recognized_object})

if __name__ == '__main__':
    app.run(debug=True)

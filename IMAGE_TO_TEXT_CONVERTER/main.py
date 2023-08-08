# Import required libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib
import pickle
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from flask_cors import CORS
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from joblib import load

# Create a Flask web application
app = Flask(__name__)
CORS(app)

# Define paths for image uploads and scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(script_dir, 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Define routes

# Home page route
@app.route('/')
def home():
    return render_template('index_2.html')

# Route for captioning uploaded images
@app.route('/caption', methods=['POST'])
def caption():
    # Process uploaded image
    img_size = 224
    uploaded_image = request.files['imageUpload']

    def read_image(path, img_size=224):
        # Load and preprocess the image
        img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img / 255.
        return img

    image_path = os.path.join(script_dir, 'uploads', 'uploaded_image.jpg')
    uploaded_image.save(image_path)
    image = read_image(image_path)
    plt.imshow(image)

    # Load models and perform captioning
    tokenizer_path = os.path.join(script_dir, 'tokenizer.pkl')
    DenseNet201_path = os.path.join(script_dir, 'DenseNet201.pkl')

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(DenseNet201_path, 'rb') as f:
        DenseNet201 = load(f)

    def feature_extraction(image_path, model1, img_size):
        # Extract image features
        img = load_img(os.path.join(image_path), target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        feature = model1.predict(img, verbose=0)
        return feature

    features = feature_extraction(image_path, DenseNet201, img_size)
    model_path = os.path.join(script_dir, 'model.h5')
    end_model = tf.keras.models.load_model(model_path)

    def idx_to_word(integer, tokenizer):
        # Convert index to corresponding word
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(model, tokenizer, max_length, feature):
        # Generate a caption for the image
        in_text = "startseq"
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length)
            y_pred = model.predict([feature, sequence])
            y_pred = np.argmax(y_pred)
            word = idx_to_word(y_pred, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

    max_length = 34
    caption_text = predict_caption(end_model, tokenizer, max_length, features)
    caption_text = caption_text.replace("startseq", "").replace("endseq", "")

    return jsonify({'caption': caption_text})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

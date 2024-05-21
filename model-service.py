from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
from lib_ml_remla import Preprocess


app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5') 

@app.route('/predict/', methods=['POST'])
def predict():
    input = request.json['input']
    padded = Preprocess.preprocess([input])
    prediction = model.predict(padded)
    return str(prediction[0][0])
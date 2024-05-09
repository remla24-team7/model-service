from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5') 

#consider saving tokenizer to dvc instead of importing entire dataset
tokenizer = load('tokenizer.joblib')


# Preprocess the input
def preprocess(text): #preprocess in ml-lib
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=200)
    return padded


@app.route('/predict/', methods=['POST'])
def predict():
    input = request.json['input']
    padded = preprocess([input])
    prediction = model.predict(padded)
    return str(prediction[0][0])
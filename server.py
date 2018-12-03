from flask import Flask
from flask import request, jsonify
from redis import Redis
from rq import Queue
from predict import make_prediction
import time

from tensorflow import get_default_graph

from keras import backend as k 
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import pandas as pd
import numpy as np
import cv2
import os
import pickle
import timeit
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load the model
print('Loading the model ...')
global model 
model = load_model('all-dogbreeds_model.h5')
print('The model is loaded')


img_width=299
img_height=299

def make_prediction(img_url):
    response = requests.get(img_url)
    img_to_predict = Image.open(BytesIO(response.content))
    img_to_predict = img_to_predict.resize((299, 299))
    image_array = img_to_array(img_to_predict)
    image_array = image_array / 255
    image_array = np.expand_dims(image_array, axis=0)
    time_start = timeit.default_timer()

    prediction = model.predict(image_array)
    highest_breed = prediction[0].argmax()
    # Load the decoder from the decoder.p file
    decoder = pickle.load(open("decoder.p", "rb"))['decoder']
    breed_prediction = decoder.inverse_transform(highest_breed)
    confidence = prediction[0][highest_breed]

    time_end = timeit.default_timer()
    pred_time = time_end - time_start

    return breed_prediction, str(round(confidence, 2)), str(round(pred_time, 1))

@app.route('/predict-dog/', methods=['GET'])
def predict():

    img_url = request.json['url']

    # Make the prediction
    breed_prediction, confidence, pred_time = make_prediction(img_url)

    result = {'breed' : breed_prediction,
              'confidence' : confidence,
              'prediction_time' : pred_time}
    
    return jsonify(result)
    # Return the result as JSON file
    # return '''<h2>Dog Breed prediction</h2>
    #           <h3>The breed of the dog is {}</h3>
    #           <h3>With a confidence of {}%</h3>
    #           <h3>Prediction executed in {} seconds</h3>'''.format(breed_prediction, confidence, pred_time)



if __name__ == '__main__':
   app.run(debug = True, use_reloader=False)
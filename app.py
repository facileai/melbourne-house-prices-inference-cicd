from flask import Flask, request, jsonify
import os

import pickle
import pandas as pd
import time
import json
import numpy as np


app = Flask(__name__)


PREDICTION_URI = os.getenv('PREDICTIONS_URI')
MODEL_NAME = os.getenv('MODEL_NAME') 
PROC_FILENAME = os.getenv('PROC_FILENAME') 

print('Load the model...')
  
with open(MODEL_NAME, 'rb') as model_file:
    model = pickle.load(model_file)

print('Load the data processor...')

with open(PROC_FILENAME, 'rb') as preproc_file:
    preproc = pickle.load(preproc_file)

column_names = list(preproc.x_names)

@app.route('/health', methods=['GET'])
def health():
    return {}, 200
    
@app.route('/bulk', methods=['POST'])
def bulk():
    uri = request.get_json().get('data_uri')

    print('open the data....')
    X_new = pd.read_csv(uri,low_memory=False)

    print('process the data...')
    new_preproc = preproc.train.new(X_new)
    new_preproc.process()
    X_new_proc = new_preproc.train.xs

    print('predict from the data...')
    y_hat = model.predict(X_new_proc)

    X_new['predicted'] = y_hat
    
    prediction_file_name = '{}/{}.csv'.format(PREDICTION_URI,int(time.time()))
    X_new.to_csv(prediction_file_name,index=False)

    print('return the response to the client...')
    out = {'location': prediction_file_name}
    
    return out, 200

    
@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json().get('observations')
    
    X_new = pd.DataFrame(data=data, columns=column_names)

    print('process the data...')
    print(X_new.shape)
    new_preproc = preproc.train.new(X_new)
    new_preproc.process()
    X_new_proc = new_preproc.train.xs

    print('predict from the data...')
    y_hat = model.predict(X_new_proc).tolist()
    print('return the response to the client...')
    out = {'timestamp': int(time.time()), 'prediction':y_hat}
    return out, 200    

if __name__ == '__main__':
    app.run(debug=True)
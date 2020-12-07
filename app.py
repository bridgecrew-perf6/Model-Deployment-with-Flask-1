from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd 
import pickle
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
model1=load_model('model009.h5')
@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    finaldf=pd.read_csv('finaldf.csv')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(finaldf['CleanTextAfterRemoveStopword'])
    test_final = tokenizer.texts_to_sequences(features)
    padded_texts = pad_sequences(test_final, maxlen=352, value=0.0)
    pred = model1.predict(padded_texts)
    list=['Group_0','Group_12','Group_24','Group_8','Group_9']
    prediction=list[np.argmax(pred)]
  
    return render_template('result.html', prediction = '{}'.format(prediction))

if __name__ == '__main__':
	app.run(debug=True)

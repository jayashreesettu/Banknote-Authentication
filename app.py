import uvicorn
from fastapi import FastAPI
from Banknote import Banknote
import pickle
import pandas as pd           
import numpy as np

app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.post('/predict')
def predict_banknote(data: Banknote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    #print(classifer.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0]> 0.5:
        prediction = 'The banknote is not authentic'
    else:
        prediction = 'The banknote is authentic'
    return {'prediction': prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
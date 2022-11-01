from flask import Flask
from flask import Flask, render_template
from flask import request
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from module import MODEL, pre_treat

### pre_load ###
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = MODEL("/Users/damon/Documents/pre_onboarding/assignment3/checkpoint","klue/roberta-base")

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def Train():
    if request.method == 'POST':
        sentence1 = pre_treat(request.form['Sentence1'])
        sentence2 = pre_treat(request.form['Sentence2'])

        output = model.forward(request.form['Sentence1'],request.form['Sentence2'])
        
        #predition
        pred = output[0][0][0].detach().numpy()
        label = '두 문장이 유사합니다.' if pred>=3.0 else '두 문장이 유사하지 않습니다.'

    return render_template('result.html',data=label, data1=pred)

if __name__ == '__main__':
    app.run(debug=True)
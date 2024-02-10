import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template,request
app=Flask(__name__)

data=pd.read_csv('./cleaned_data.csv')
pipe= pickle.load(open("./RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations=sorted(data['Address'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('Address')
    bhk=request.form.get('BHK')
    bath=request.form.get('Bathrooms')
    sqft=request.form.get('Total_sqft')
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['Address','Total_sqft','Bathrooms','BHK'])
    prediction=pipe.predict(input)[0]
    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True,port=5500)
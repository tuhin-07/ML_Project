from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData , PredictPipeline


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )

        data_frame = data.get_data_as_data_frame()

        Predict_Pipeline = PredictPipeline()

        predicted_Data = Predict_Pipeline.predict(data_frame)

        return render_template('home.html',results= predicted_Data[0])
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
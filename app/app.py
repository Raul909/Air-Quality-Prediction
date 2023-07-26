from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('D:\\Air Quality Prediction\\models\\random_forest_model.pkl')

# Load the list of features used during training (including 'City' columns)
with open('D:\\Air Quality Prediction\\app\\feature_list.txt', 'r') as f:
    feature_list = f.read().splitlines()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input data from the form
        city = request.form['city']
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])

        # Create a DataFrame from user input data
        user_input = pd.DataFrame({
            'City_' + city: [1],  # Set the selected city to 1, others to 0
            'PM2.5': [pm25],
            'PM10': [pm10],
        })

        # Add other input fields as per your model features with a default value of 0
        for feature in feature_list:
            if feature not in user_input.columns:
                user_input[feature] = 0

        # Predict the AQI Bucket with the model
        aqi_bucket_prediction = model.predict(user_input)[0]

        return render_template('result.html',
                               city=city,
                               pm25=pm25,
                               pm10=pm10,
                               aqi_bucket=aqi_bucket_prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

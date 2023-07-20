from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Get the absolute path to the 'data' folder
data_file = 'D:\\Air Quality Prediction\\src\\data\\city_day.csv'
input_data = pd.read_csv(data_file)

# Perform one-hot encoding on the 'City' column
input_data = pd.get_dummies(input_data, columns=['City'])

app = Flask(__name__)

# Load the trained models
random_forest_model = joblib.load('D:\\Air Quality Prediction\\models\\random_forest_model.pkl')
linear_regression_model = joblib.load('D:\\Air Quality Prediction\\models\\linear_regression_model.pkl')

# Rest of the code remains unchanged...


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input data from the form
        city = request.form['city']
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        # Add other input fields as per your model features

        # Create a DataFrame from user input data
        user_input = pd.DataFrame({
            'PM2.5': [pm25],
            'PM10': [pm10],
            # Include the one-hot encoded columns for all cities with default value 0
            **{col: [0] for col in input_data.columns if col.startswith('City_')},
        })

        # Set the value to 1 for the corresponding city in the user input DataFrame
        user_input['City_' + city] = 1

        # Predict air quality levels with the Random Forest model
        random_forest_prediction = random_forest_model.predict(user_input)

        # Predict air quality levels with the Linear Regression model
        linear_regression_prediction = linear_regression_model.predict(user_input)

        return render_template('result.html',
                               city=city,
                               random_forest_prediction=random_forest_prediction[0],
                               linear_regression_prediction=linear_regression_prediction[0])
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from logger import logging

app = Flask(__name__)

logging.info('Flask server started')

# Load the trained logistic regression model and MinMaxScaler
with open('static/model/model.pickle', 'rb') as file:
    logistic_model = pickle.load(file)

with open('static/model/scaler.pickle', 'rb') as scaler_file:
    norm = pickle.load(scaler_file)


# Define the prediction function
def make_single_prediction(model, scaler, input_features):
  # Create a DataFrame with the input features
  input_data = pd.DataFrame(input_features, index=[0])

  # Ensure the order of columns matches the model's expectations
  input_data = input_data[['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

  # Normalize the input data using the same MinMaxScaler
  input_data_normalized = scaler.transform(input_data)

  # Make a prediction
  prediction = model.predict(input_data_normalized)

  # Map predicted value to labels
  prediction_label = 'High Quality Wine' if prediction[0] == 1 else 'Low Quality Wine'

  return prediction_label


def create_feature_chart(input_features):
    # Create a bar chart for input feature values
    fig = px.bar(x=list(input_features.keys()), y=list(input_features.values()), labels={'x': 'Feature', 'y': 'Value'})
    graph_json = fig.to_json()
    return graph_json


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_inputs = {
            'type': int(request.form['type']),
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }

        prediction = make_single_prediction(logistic_model, norm, user_inputs)

        # Create a bar chart for input feature values
        graph_json = create_feature_chart(user_inputs)

        return render_template("index.html", prediction=prediction, graph_json=graph_json)

    return render_template("index.html", prediction=None, graph_json=None)



if __name__ == "__main__":
    app.run(debug=True)
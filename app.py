from flask import Flask, render_template, request
import pickle
import pandas as pd
import plotly.express as px
# import plotly.figure_factory as ff
# import numpy as np
import plotly.graph_objs as go
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from sklearn.metrics import roc_auc_score
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

# def create_correlation_heatmap(corr_matrix, column_names):
#     fig = ff.create_annotated_heatmap(z=corr_matrix, x=column_names, y=column_names)
#     graph_json = fig.to_json()
#     return graph_json

# def create_scatter_plot(data, x_feature, y_feature, color_feature):
#     fig = px.scatter(data, x=x_feature, y=y_feature, color=color_feature)
#     graph_json = fig.to_json()
#     return graph_json

# def create_model_metrics(y_true, y_pred_prob):
#     auc_score = roc_auc_score(y_true, y_pred_prob) 

#     # Convert the list of probabilities to a list of 0s and 1s based on a threshold
#     threshold = 0.5
#     y_pred = [1 if prob > threshold else 0 for prob in y_pred_prob]

#     cm = confusion_matrix(y_true, y_pred)
#     report = classification_report(y_true, y_pred)

#     return {
#         'auc_score': auc_score,
#         'confusion_matrix': cm,
#         'classification_report': report
#     }

def create_line_chart(input_data):
    x_values = list(input_data.keys())
    y_values = list(input_data.values())

    trace = go.Scatter(x=x_values, y=y_values, mode='lines+markers', name='User Inputs')
    data = [trace]
    layout = go.Layout(xaxis=dict(title='Features'), yaxis=dict(title='Values'))
    fig = go.Figure(data=data, layout=layout)
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

        # Sample dataset for correlation heatmap and scatter plot
        # data = pd.DataFrame(user_inputs, index=[0])

        # correlation_matrix = data.corr().to_numpy()
        # column_names = data.columns.tolist()

        prediction = make_single_prediction(logistic_model, norm, user_inputs)

        # Create a bar chart for input feature values
        graph_json = create_feature_chart(user_inputs)

        # Create a correlation heatmap
        # correlation_heatmap = create_correlation_heatmap(correlation_matrix, column_names)

        # Create a scatter plot
        # scatter_plot = create_scatter_plot(data, x_feature='alcohol', y_feature='fixed acidity', color_feature='type')

        # # Sample model metrics
        # y_true = [0, 1, 1, 0, 0]  # True labels
        # y_pred_prob = [0.2, 0.8, 0.7, 0.3, 0.1]  # Predicted probabilities

        # model_metrics = create_model_metrics(y_true, y_pred_prob)

        # Create a line chart
        line_chart = create_line_chart(user_inputs)

        return render_template("index.html", prediction=prediction, graph_json=graph_json, line_chart=line_chart)

    return render_template("index.html", prediction=None, graph_json=None, line_chart = None)


if __name__ == "__main__":
    app.run(debug=True)
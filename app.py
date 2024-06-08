"""
app.py

Purpose:
This file creates a Flask web application that handles user input, processes it using the trained model, and returns the prediction result. It also provides an API endpoint for making predictions.

Imports:
- Flask, request, render_template, jsonify from flask: Flask is used to create the web application. request is used to handle incoming HTTP requests. render_template is used to render HTML templates. jsonify is used to return JSON responses.
- CORS, cross_origin from flask_cors: CORS (Cross-Origin Resource Sharing) support.
- CustomData, PredictPipeline from src.pipeline.predict_pipeline: Custom classes to handle input data and the prediction process.
"""

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a Flask application instance
application = Flask(__name__)

# Alias for the application instance
app = application

# Enable CORS support
CORS(app)

@app.route('/')
@cross_origin()
def home_page():
    """
    Render the home page.

    Returns:
    HTML template for the home page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    """
    Handle the data prediction process.

    Returns:
    HTML template with the prediction results for POST requests.
    HTML template for the home page for GET requests.
    """
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Create an instance of CustomData with input data from the form
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        # Convert the custom data to a DataFrame
        pred_df = data.get_data_as_dataframe()
        
        print(pred_df)

        # Create an instance of PredictPipeline and predict the results
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0], 2)
        return render_template('index.html', results=results, pred_df=pred_df)

@app.route('/predictAPI', methods=['POST'])
@cross_origin()
def predict_api():
    """
    API endpoint for making predictions.

    Returns:
    JSON response with the prediction result.
    """
    if request.method == 'POST':
        # Create an instance of CustomData with input data from the JSON request
        data = CustomData(
            carat = float(request.json['carat']),
            depth = float(request.json['depth']),
            table = float(request.json['table']),
            x = float(request.json['x']),
            y = float(request.json['y']),
            z = float(request.json['z']),
            cut = request.json['cut'],
            color = request.json['color'],
            clarity = request.json['clarity']
        )

        # Convert the custom data to a DataFrame
        pred_df = data.get_data_as_dataframe()
        
        # Create an instance of PredictPipeline and predict the results
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        # Return the prediction result as a JSON response
        dct = {'price': round(pred[0], 2)}
        return jsonify(dct)

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=8000)


# Import necessary libraries
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flash application
superkart_predication_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model
# model_path = "backend_files/superkart_model_v1_0.joblib"
model = joblib.load("superkart_model_v1_0.joblib")

# Define a route for the homepage (GET request)
@superkart_predication_api.route("/")
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "<h1>SuperKart Sales Predictor API</h1>"

# Define an endpoint for single product sales forecast
@superkart_predication_api.post("/v1/predict")
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint
    It expects a JSON payload containing product details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        "Product_Weight": product_data["Product_Weight"],
        "Product_Sugar_Content": product_data["Product_Sugar_Content"],
        "Product_Allocated_Area": product_data["Product_Allocated_Area"],
        "Product_MRP": product_data["Product_MRP"],
        "Store_Size": product_data["Store_Size"],
        "Store_Location_City_Type": product_data["Store_Location_City_Type"],
        "Store_Type": product_data["Store_Type"],
        "Product_Id_char": product_data["Product_Id_char"],
        "Store_Age_Years": product_data["Store_Age_Years"],
        "Product_Type_Category": product_data["Product_Type_Category"]
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # First, convert categorical columns (object type) to 'category' type 
    # to match model's training expectations
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].astype('category')

    # Make predictions using the loaded model
    prediction = model.predict(input_data)[0]

    # Convert predicted total to Python float
    predicted_sales = round(float(prediction), 2)

    # Return the predicted sales as a JSON response
    return jsonify({"Predicted Sales (in dollars)": predicted_sales})

# Define an endpoint for batch prediction (POST request)
@superkart_predication_api.post("/v1/predictbatch")
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint
    It expects a CSV file containing product details for multiple products
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file_data = request.files["file"]

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file_data)

    # First, convert categorical columns (object type) to 'category' type 
    # to match model's training expectations
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].astype('category')

    # Make predictions for all products in DataFrame
    predictions = model.predict(input_data).tolist()

    # Calculate actual sales
    predicted_sales = [round(float((predictions), 2)) for sale in predictions]

    # Create 0-based row numbers as strings
    row_nums = np.arange(len(input_data)).astype(str)

    # Product_Id_chars as a list
    product_ids = input_data["Product_Id_char"].tolist()

    # Combine product_ids and row_num and as keys
    product_keys = np.char.add(product_ids, row_nums)
    product_keys = product_keys.tolist()

    # Create a dictionary with product_ids as keys and predicted sales as values
    predictions_dict = dict(zip(product_keys, predicted_sales))

    # Return the predictions as a JSON response
    return jsonify(predictions_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == "__main__":
    superkart_predication_api.run(debug = True)

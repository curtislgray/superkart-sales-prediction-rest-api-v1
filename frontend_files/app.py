import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://localhost:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Predictor v1.0")

# Section for the online prediction
st.subheader("Online Sales Prediction")

# Collect user input for product features
product_weight = st.number_input("Product Weight", min_value = 0.01, max_value = 1000.00, step = 1.00, value = 10.00)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar","Regular","No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value = 0.001, max_value = 1.000, step = 0.001, value = 0.027)
product_mrp = st.number_input("Product MRP)", min_value = 0.000, max_value = 1000.000, step = 11.000, value = 50.000)
store_size = st.selectbox("Store Size", ["High","Medium","Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2","Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
product_id_char = st.selectbox("Product ID", ["FD", "DR","NC"])
store_age_years = st.number_input("Store Age (in years)", min_value = 0, max_value = 400, step = 1, value = 10)
product_type_category = st.selectbox("Product Type", ["Perishables","Non Perishables"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    "Product_Weight": product_weight, # 12.66,
    "Product_Sugar_Content": product_sugar_content, # "Low Sugar",
    "Product_Allocated_Area": product_allocated_area, # 0.027,
    "Product_MRP": product_mrp, # 117.08,
    "Store_Size": store_size, # "Medium",
    "Store_Location_City_Type": store_location_city_type, # "Tier 2",
    "Store_Type": store_type, # "Supermarket Type2",
    "Product_Id_char": product_id_char, # "FD",
    "Store_Age_Years": store_age_years,  # 16,
    "Product_Type_Category": "Non Perishables"
}])

# Make a prediction when the "Predict" button is clicked
if st.button("Predict", type = "primary"):
  response = requests.post(f"{BACKEND_URL}/v1/predict", json = input_data.to_dict(orient = "records")[0])

  # Check if the request was successful
  if response.status_code == 200:
    prediction = response.json()['Predicted Sales (in dollars)']
    st.success(f"Predicted Sales: ${prediction}")
  else:
    st.error("Error getting prediction.")

# Section for Batch Predictions
st.subheader("Batch Sales Prediction")

# Allow users to upload a CSV file for batch predictions
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions ...", type = ["csv"])

# Make a batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
  if st.button("Predict Batch", type = "primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files = {"file": uploaded_file})
    if response.status_code == 200:
      predictions = response.json()
      st.success("Batch predictions successful!")
      st.write(predictions)
    else:
      st.error("Error getting batch predictions.")


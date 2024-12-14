import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Machine Learning Playground")

# File uploader for datasets
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    
    # Clean the `Weekly_Sales` column (remove commas and convert to float)
    data["Weekly_Sales"] = data["Weekly_Sales"].replace(',', '', regex=True).astype(float)
    
    # Display the data
    st.write("Data preview", data.head())
    
    # Show basic stats
    st.write("Basic statistics:", data.describe())

    # Train a simple regression model
    if st.button("Train Model"):
        st.write("Training model...")

        # Define features (X) and target (y)
        X = data.drop(columns=["Weekly_Sales"])  # Features (all columns except target)
        y = data["Weekly_Sales"]  # Target column

        # Handle categorical variables if present
        X = pd.get_dummies(X, drop_first=True)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the regression model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Evaluate the model
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("R-squared Score:", r2_score(y_test, y_pred))

        # Plot predicted vs actual values
        st.write("Predicted vs Actual:")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Actual Weekly Sales")
        plt.ylabel("Predicted Weekly Sales")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)

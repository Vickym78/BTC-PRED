import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

# Streamlit App Title
st.title("Bitcoin Price Analysis and Prediction")

# Load the dataset directly from a local file
file_path = 'BTC-USD.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Plot Bitcoin daily closing prices over time
st.subheader("Bitcoin Daily Closing Prices")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Date', y='Close', alpha=0.8, color='orange')
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title("Bitcoin Daily Closing Prices")
plt.grid(True)
st.pyplot(plt)

# Plot a histogram of Bitcoin daily closing prices
st.subheader("Histogram of Bitcoin Closing Prices")
plt.figure(figsize=(12, 8))
plt.hist(df['Close'], bins=50, edgecolor='k', color='yellow')
plt.title('Bitcoin Daily Closing Price Histogram')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.grid(True)
st.pyplot(plt)

# Prepare data for prediction
projection_bitcoin = 14  # Number of days to predict
df['Prediction'] = df[['Close']].shift(-projection_bitcoin)

# Features and target
X_Bitcoin = np.array(df[['Close']])[:-projection_bitcoin]
y_Bitcoin = df['Prediction'].values[:-projection_bitcoin]

# Split data into training and testing sets
x_train_Bitcoin, x_test_Bitcoin, y_train_Bitcoin, y_test_Bitcoin = train_test_split(
    X_Bitcoin, y_Bitcoin, test_size=0.20
)

# Train Linear Regression model
linReg_Bitcoin = LinearRegression()
linReg_Bitcoin.fit(x_train_Bitcoin, y_train_Bitcoin)

# Display model coefficients
w = linReg_Bitcoin.coef_[0].round(4)
b = linReg_Bitcoin.intercept_.round(2)
st.subheader("Linear Regression Model")
st.write(f"Model Coefficient (w): {w}")
st.write(f"Model Intercept (b): {b}")

# Prepare data for predictions
x_projection_Bitcoin = np.array(df[['Close']])[-projection_bitcoin:]
linReg_prediction_Bitcoin = linReg_Bitcoin.predict(x_projection_Bitcoin)

# Model confidence
linReg_confidence_Bitcoin = linReg_Bitcoin.score(x_test_Bitcoin, y_test_Bitcoin)
st.write(f"Model Confidence: {round(linReg_confidence_Bitcoin * 100, 2)}%")

# Plot predicted vs actual prices
st.subheader("Predicted vs Actual Prices")
plt.figure(figsize=(12, 8))
plt.plot(linReg_prediction_Bitcoin, label='Predicted Price', color='blue')
plt.plot(x_projection_Bitcoin, label='Actual Price', color='orange')
plt.title('Price Prediction vs Actual Price')
plt.xlabel('Days Ahead')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.grid(True)
st.pyplot(plt)

# Show prediction values
st.subheader("Predicted Prices for the Next 14 Days")
st.write(pd.DataFrame({"Predicted Price": linReg_prediction_Bitcoin}))

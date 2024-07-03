# LSTM-price
Replicating the functionality of a Stock Broker in predicting the future prices of particular stocks using LSTM

## Overview
This repository contains Python code for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The data is acquired from the Tiingo API, and TensorFlow is used for building and training the LSTM model. Based on the performance history of a stock (based on factors such as open, close price, high, low and volume traded), predicting its future price.

## Features
  1. **Data Acquisition:** Historical stock price data is fetched using the Tiingo API.
  2. **Data Preprocessing:** Features including 'open', 'high', 'low', 'close', and 'volume' are selected and normalized using Min-Max scaling.
  3. **Sequence Creation:** Sequences of historical data are created to predict future stock prices.
  4. **Model Building:** A deep LSTM model is designed using TensorFlow/Keras to learn from historical patterns and predict future stock prices.
  5. **Model Evaluation:** The model is evaluated on a test set, and performance metrics such as Mean Squared Error (MSE) are calculated.
  6. **Visualization:** Predicted vs. actual stock prices are visualized using matplotlib.

## Setup Instructions
  1. Clone the repository
  2. Obtain an API key from Tiingo (replace YOUR_API_KEY with your actual API key) and set it in `main.py`
  3. Run the Data Acquisition script
     `python dataAcquisition.py`
  4. Run the main script
     `python main.py`

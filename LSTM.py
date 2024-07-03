import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = df[['date', 'open', 'high', 'low', 'close', 'volume']]     # Feature Selection

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])   # Data Normalization

def create_sequences(data, seq_length):                          # Sequence Creation
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length][3])  # Using the 'close' price as the target
    return np.array(sequences), np.array(targets)

seq_length = 75
X, y = create_sequences(scaled_data, seq_length)

# Train-Test-Validation Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
# LSTM Model Design
model = Sequential()    
model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 5)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mean_squared_error')  # Compile the model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Train the model

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

loss = model.evaluate(X_test, y_test)   # Model Evaluation
print(f'Test Loss: {loss}')

import matplotlib.pyplot as plt     # Predicting and plotting

y_pred = model.predict(X_test)
y_pred_unscaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 4)), y_pred), axis=1))[:, -1]
y_test_unscaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)), axis=1))[:, -1]

plt.figure(figsize=(14, 5))
plt.plot(y_test_unscaled, label='True Price')
plt.plot(y_pred_unscaled, label='Predicted Price')
plt.legend()
plt.show()

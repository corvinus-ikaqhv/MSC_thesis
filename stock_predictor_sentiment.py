import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

seq_len = 24
scaler = StandardScaler()

# Loading the historical stock prices data
#df = pd.read_csv('MSFT_jan.csv',delimiter=';')
df = pd.read_csv('MSFT_jan_enriched.csv',delimiter=';')
data = df.drop(['Date'], axis=1)

# Standardizing each column in the dataset
df_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
targets = df_standardized['StockPrice'].values

# Reshaping data to 3D array with shape (num_samples, seq_len, num_features)
num_samples = data.shape[0] - seq_len
num_features = data.shape[1]
X = np.zeros((num_samples, seq_len, num_features))
y = np.zeros(num_samples)
for i in range(num_samples):
    X[i] = df_standardized[i:i+seq_len]
    y[i] = targets[i+seq_len]

# Splitting the data into training and testing sets
train_size = int(len(X) * 0.7)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# unit = complexity of weight matrix
# Building the LSTM model
#model = Sequential()
#model.add(LSTM(units=20, return_sequences=True, input_shape=(seq_len, num_features)))
#model.add(LSTM(units=10, return_sequences=False))
#model.add(Dropout(0,5))
#model.add(Dense(units=1))

model = Sequential()
model.add(LSTM(units=400, return_sequences=True, input_shape=(seq_len, num_features)))
model.add(LSTM(units=300, return_sequences=False))
model.add(Dropout(0,5))
model.add(Dense(units=1))


# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=120, batch_size=32)

# Making predictions on the testing data set
y_pred = model.predict(X_test)
scaler.fit(data[['StockPrice']])
y_pred = scaler.inverse_transform(y_pred)

# Getting the last sequence in X_test
last_sequence = X_test[-1:]
# Using the model to generate a prediction based on the last sequence
predicted_price = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price)

# Plotting the predicted and actual stock prices
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test.reshape(y_test.shape[0], -1)), label="Actual")
plt.plot(y_pred, label="Predicted")
plt.plot(len(y_pred), predicted_price[0][0], "rx", label="Predicted Price")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time (hours)")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("C:/Users/Laura/Thesis/LSTM/actual_vs_predicted_stock_prices.png")
plt.show()

mse = model.evaluate(X_test, y_test)
print(mse)

# Print the predicted price
print(f"Predicted stock price based on last sequence: {predicted_price[0][0]}")

from preprocessing import load_and_preprocess
from model import create_sequences, build_model, scale_data
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = load_and_preprocess("../data/consumption.csv", "../data/weather_hourly.csv")

# Scale data
scaled, scaler = scale_data(df)

# Create sequences
X, y = create_sequences(scaled, 24)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Train model
es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[es]
)

# Predictions
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# Plot graph
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Energy Consumption Prediction")
plt.show()

# Save model
model.save("../model_energy.h5")

print("✅ Training Complete with Graph + Metrics!")

from preprocessing import load_and_preprocess
from model import create_sequences, build_model, scale_data
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data
df = load_and_preprocess("../data/consumption.csv", "../data/weather_hourly.csv")

# Scale
scaled, scaler = scale_data(df)

# Sequences
X, y = create_sequences(scaled, 24)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Train
es = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[es]
)

# Predictions
predictions = model.predict(X_test)

# Plot results
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Energy Consumption Prediction")
plt.show()

# Save model
model.save("../model_energy.h5")

print("✅ Training Complete & Graph Generated!")

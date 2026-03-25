from preprocessing import load_and_preprocess
from model import create_sequences, build_model, scale_data
from tensorflow.keras.callbacks import EarlyStopping

df = load_and_preprocess("../data/consumption.csv", "../data/weather_hourly.csv")

scaled, scaler = scale_data(df)

X, y = create_sequences(scaled, 24)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = build_model((X_train.shape[1], X_train.shape[2]))

es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=20,
          batch_size=32,
          callbacks=[es])

model.save("../model_energy.h5")

print("Done ✅")

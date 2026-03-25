import pandas as pd

def load_and_preprocess(consumption_file, weather_file):
    df = pd.read_csv(consumption_file, parse_dates=["timestamp"])
    weather = pd.read_csv(weather_file, parse_dates=["timestamp"])

    df.set_index("timestamp", inplace=True)
    weather.set_index("timestamp", inplace=True)

    df = df.resample("1H").sum()
    df["temp"] = weather["temp"].resample("1H").mean()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    for lag in [1,24,48]:
        df[f"lag_{lag}"] = df["consumption_kwh"].shift(lag)

    df.dropna(inplace=True)
    return df

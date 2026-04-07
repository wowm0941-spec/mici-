# app.py - KI Index Streamlit App
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mein KI Index", layout="wide")
st.title("🤖 Mein KI Index (Streamlit)")

# --- Sidebar: Einstellungen ---
st.sidebar.header("Einstellungen")
tickers_input = st.sidebar.text_input("Tickers (Komma-getrennt)", "AAPL,MSFT,AMZN,NVDA,GOOGL,META,TSLA")
start_date = st.sidebar.date_input("Startdatum", pd.to_datetime("2020-01-01"))
train_fraction = st.sidebar.slider("Train/Test Split (Anteil Training)", 0.6, 0.95, 0.8)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) < 2:
    st.error("Bitte mindestens 2 Ticker eingeben.")
    st.stop()

st.info(f"Lade Daten für: {', '.join(tickers)}. Das kann ein paar Sekunden dauern.")

# --- Daten laden ---
@st.cache_data(ttl=3600)
def load_data(tickers, start):
    df = yf.download(tickers, start=start)["Close"]
    return df.dropna()

data = load_data(tickers, start_date)

if data.empty:
    st.error("Keine Daten geladen. Prüfe die Ticker oder das Startdatum.")
    st.stop()

st.write("Daten geladen:", data.shape[0], "Zeilen,", data.shape[1], "Spalten")

# --- Feature Engineering (pro Aktie) ---
def build_dataset(data):
    features = []
    for stock in data.columns:
        s = data[stock]
        df = pd.DataFrame(index=s.index)
        # einfache Features
        df["momentum_1m"] = s.pct_change(20)
        df["momentum_3m"] = s.pct_change(60)
        df["volatility_1m"] = s.pct_change().rolling(20).std()
        df["target_1m"] = s.pct_change(20).shift(-20)  # Ziel: Rendite in 1 Monat
        df["stock"] = stock
        features.append(df.dropna())
    return pd.concat(features)

dataset = build_dataset(data)
st.write("Feature-Datensatz Größe:", dataset.shape)

if dataset.empty:
    st.error("Feature-Datensatz leer. Mehr Daten oder anderes Startdatum wählen.")
    st.stop()

# --- Train/Test split ---
def train_model(dataset, train_frac=0.8):
    df = dataset.copy()
    # Shuffle nach Zeit vermeiden -> gruppieren nach Zeit nicht nötig, wir splitten per Index
    cutoff = int(len(df) * train_frac)
    X = df[["momentum_1m", "momentum_3m", "volatility_1m"]]
    y = df["target_1m"]
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

with st.spinner("Modell trainieren..."):
    model, X_test, y_test = train_model(dataset, train_fraction)

st.success("Modell trainiert")

# --- Aktuelle Features für Vorhersage (letzte verfügbare Werte) ---
def build_latest_features(data):
    rows = []
    for stock in data.columns:
        s = data[stock]
        latest = {}
        latest["stock"] = stock
        latest["momentum_1m"] = s.pct_change(20).iloc[-1]
        latest["momentum_3m"] = s.pct_change(60).iloc[-1]
        latest["volatility_1m"] = s.pct_change().rolling(20).std().iloc[-1]
        rows.append(latest)
    return pd.DataFrame(rows).set_index("stock")

latest_df = build_latest_features(data)
st.write("Aktuelle Features:", latest_df)

# Vorhersage
preds = model.predict(latest_df[["momentum_1m", "momentum_3m", "volatility_1m"]])
pred_series = pd.Series(preds, index=latest_df.index, name="predicted_return")

# Negative Vorhersagen clippen (optional)
pred_series = pred_series.clip(lower=0)

# Softmax Gewichte
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

weights = pd.Series(softmax(pred_series.values), index=pred_series.index)
st.write("📊 Gewichte (KI):")
st.dataframe(weights.to_frame("weight"))

# Index berechnen (Basis 1000)
normalized = data / data.iloc[0]
index_series = (normalized * weights).sum(axis=1) * 1000

# Anzeige
st.subheader("Index-Verlauf")
st.line_chart(index_series)

st.subheader("Letzte Werte")
col1, col2 = st.columns(2)
col1.write("Letzter Index-Wert:")
col1.metric("Index", f"{index_series.iloc[-1]:.2f}")
col2.write("Gewichte")
col2.dataframe(weights.to_frame("weight"))

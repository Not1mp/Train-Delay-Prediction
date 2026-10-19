# Indian Train Delay Predictor

A small web app that predicts how late an Indian train will be at a station you choose.

Enter a train number and a destination, click **Predict delay**, and the app shows the expected delay and arrival time.

## How it works

The app uses two models:

- **Train not started (or no live data):** predicts from past delays of that train at that station.
- **Train already running:** uses the train's live delay from the [RailRadar](https://api.railradar.in) API to make a better prediction.

## Files

| File | What it does |
|------|--------------|
| `train_model.py` | Trains the models and saves them to `artifacts.joblib` |
| `app.py` | The Streamlit web app |
| `live_status.py` | Gets live train status from RailRadar |

You also need these data files in the same folder:

- `train_routes_delays_Sep2024.csv`
- `train_routes_Sep2024.csv`
- `stations_zones_mapping.json` (optional)

## Setup

**1. Install the packages**

```bash
pip install pandas xgboost scikit-learn joblib streamlit requests
```

**2. Train the models** (run this once)

```bash
python train_model.py
```

**3. Add your RailRadar API key**

Create a file named `railradar_key.txt` in the project folder and paste your key into it.

Or put it in `.streamlit/secrets.toml`:

```toml
RAILRADAR_API_KEY = "your-key-here"
```

Don't share your key or upload it to GitHub.

**4. Start the app**

```bash
streamlit run app.py
```

## Try it without an API key

Open `live_status.py` and change `MODE = "railradar"` to `MODE = "demo"`. The app will use fake live data so you can test it.

## Using the app

1. Type a train number, like `12303`.
2. Choose a destination station.
3. Click **Predict delay**.

You will see the predicted delay, the scheduled and predicted arrival times, and the typical error of the model.

## Good to know

- Only trains in the September 2024 data are supported.
- The model learns from one month of data, so it doesn't know about seasons like monsoon or fog.
- The "typical error" is an average. Individual predictions can be off by more.
- Trains that arrive early are shown as 0 minutes late.
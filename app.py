# =====================================================================
# TRAIN DELAY APP
# First run:  python train_model.py
# Then run:   streamlit run app.py
# =====================================================================

import os
import joblib
import pandas as pd
import streamlit as st
from live_status import get_live_status

st.title("Indian Train Delay Predictor")

# The saved models are created by train_model.py
if not os.path.exists("artifacts.joblib"):
    st.error("Please run python train_model.py first. That creates artifacts.joblib.")
    st.stop()

data = joblib.load("artifacts.joblib")
model_a = data["model_a"]
model_b = data["model_b"]
hist = data["hist"]
station_avg = data["station_mean"]
routes = data["routes"]

def get_station_average(station):
    """Average delay of all trains at this station."""
    row = station_avg[station_avg["station_code"] == station]
    if len(row) > 0:
        return float(row["station_mean"].iloc[0])
    return data["gmean"]


def get_history(train_no, station):
    """Usual delay of this train at this station, and how many past days we have."""
    row = hist[(hist["train_no"] == train_no) & (hist["station_code"] == station)]
    if len(row) > 0:
        usual = row["h_sum"].iloc[0] / row["h_n"].iloc[0]
        days = int(row["h_n"].iloc[0])
        return usual, days
    return get_station_average(station), 0


def make_station_labels(route):
    """Text shown in the dropdown, like 'NDLS  (440 km)'."""
    labels = []
    for i in range(len(route)):
        code = str(route["station_code"].iloc[i])
        km = str(int(route["distance"].iloc[i]))
        labels.append(code + "  (" + km + " km)")
    return labels


def pick_label(index):
    return station_labels[index]


def predict_delay(mode, dest, last_i, current_delay):
    usual_delay, days = get_history(train_no, dest["station_code"])
    inputs = {
        "distance": dest["distance"],
        "sched_hour": dest["sched_hour"],
        "avg_here": usual_delay,
        "hist_n": days,
        "station_mean": get_station_average(dest["station_code"]),
    }

    if mode == "A":
        X = pd.DataFrame([inputs])[data["feats_a"]]
        prediction = model_a.predict(X)[0]
        typical_error = data["mae_a"]
        last_code = None
    else:
        last = route.iloc[last_i]
        last_usual, _extra = get_history(train_no, last["station_code"])
        inputs["prev_delay"] = current_delay
        inputs["prev_excess"] = current_delay - last_usual
        inputs["gap_km"] = dest["distance"] - last["distance"]
        inputs["stops_between"] = dest["route_idx"] - last["route_idx"]
        X = pd.DataFrame([inputs])[data["feats_b"]]
        prediction = model_b.predict(X)[0]
        typical_error = data["mae_b"]
        last_code = last["station_code"]

    if prediction < 0:
        prediction = 0
    return prediction, typical_error, usual_delay, days, last_code


# ---- 1) train number ----
train_no = st.text_input("Train number (example: 12303)").strip()
if train_no == "":
    st.stop()

route = routes[routes["train_no"] == train_no].reset_index(drop=True)
if len(route) == 0:
    st.error("This train is not in the September 2024 dataset.")
    st.stop()

train_name = str(data["names"].get(train_no, ""))
st.caption(train_no + " " + train_name)

station_labels = make_station_labels(route)

# ---- 2) destination ----
dest_i = st.selectbox(
    "Destination station",
    range(len(route)),
    format_func=pick_label,
    index=len(route) - 1,
)
dest = route.iloc[dest_i]

# ---- 4) predict ----
if st.button("Predict delay"):
    mode = "A"
    last_i = None
    current_delay = 0
    source = "history only (train treated as not started)"
    live = None

    with st.spinner("Checking live train status..."):
        live = get_live_status(train_no, route)

    if live is None:
        source = "no live data found (train may not have started)"
    else:
        st.info(live["message"])
        if live.get("train_name"):
            st.write("Live train name: " + str(live["train_name"]))
        if live.get("next_station"):
            st.write("Next halt: " + str(live["next_station"]))

        if live.get("ok_for_prediction") and live.get("last_station") in list(route["station_code"]):
            mode = "B"
            last_i = list(route["station_code"]).index(live["last_station"])
            current_delay = live["delay"]
            source = "RailRadar live status"
        else:
            source = "history only (live data not usable for this stop)"

    station_code = str(dest["station_code"]).upper()
    station_time = next(
        (
            item for item in (live or {}).get("station_times", [])
            if item.get("station_code") == station_code
        ),
        {},
    )
    scheduled_arrival = pd.to_datetime(
        station_time.get("scheduled_arrival"), errors="coerce"
    )

    if mode == "B" and last_i >= dest_i:
        st.info("This train has already passed " + dest["station_code"] + ".")
        if not pd.isna(scheduled_arrival):
            st.metric(
                "Scheduled arrival at " + station_code,
                scheduled_arrival.strftime("%d %b %Y, %I:%M %p IST"),
            )
        st.stop()

    prediction, typical_error, usual_delay, days, last_code = predict_delay(
        mode, dest, last_i, current_delay
    )

    if mode == "B" and last_code is not None:
        st.write(
            "Train is at "
            + str(last_code)
            + " running "
            + str(int(current_delay))
            + " min late."
        )

    st.metric("Predicted delay at " + dest["station_code"], str(round(prediction)) + " min")
    if not pd.isna(scheduled_arrival):
        st.metric(
            "Scheduled arrival at " + station_code,
            scheduled_arrival.strftime("%d %b %Y, %I:%M %p IST"),
        )
        predicted_arrival = scheduled_arrival + pd.to_timedelta(float(prediction), unit="m")
        st.metric(
            "Predicted arrival at " + station_code,
            predicted_arrival.strftime("%d %b %Y, %I:%M %p IST"),
        )
    else:
        st.write("Scheduled arrival time is unavailable from RailRadar.")
    st.write("Input used: " + source)
    st.write("Typical error on test days: about +/- " + str(round(typical_error)) + " min")
    st.write(
        "This train's usual delay at "
        + dest["station_code"]
        + ": "
        + str(round(usual_delay))
        + " min (from "
        + str(days)
        + " past days)"
    )

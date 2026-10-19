# =====================================================================
# TRAIN DELAY MODEL
# Run this first:  python train_model.py
# It reads the September 2024 railway files and saves artifacts.joblib
# =====================================================================

import json
import joblib
import pandas as pd
from xgboost import XGBRegressor

# File names in this folder
DELAY_FILE = "train_routes_delays_Sep2024.csv"
ROUTE_FILE = "train_routes_Sep2024.csv"
ZONE_FILE = "stations_zones_mapping.json"

# Last 5 days are used only for testing (the model does not train on them)
TEST_DAYS = 5

# For "train already running", look back this many stops
LOOK_BACK_STOPS = [1, 3, 6, 10]

# Numbers we give to the models
FEATURES_A = ["distance", "sched_hour", "avg_here", "hist_n", "station_mean"]
FEATURES_B = FEATURES_A + ["prev_delay", "prev_excess", "gap_km", "stops_between"]


def time_text_to_hour(time_text):
    """Turn '08:00 AM' or '01:12 PM' into an hour from 0 to 23."""
    if pd.isna(time_text):
        return None

    text = str(time_text).strip().upper()
    if text == "" or text == "NAN":
        return None

    parts = text.split()
    clock = parts[0]
    am_pm = "AM"
    if len(parts) > 1:
        am_pm = parts[1]

    hour_and_minute = clock.split(":")
    try:
        hour = int(hour_and_minute[0])
    except ValueError:
        return None

    # 12 AM is midnight (hour 0). 1 PM is hour 13.
    if am_pm == "PM" and hour != 12:
        hour = hour + 12
    if am_pm == "AM" and hour == 12:
        hour = 0
    return hour


def load_and_clean():
    """Read delay rows and add distance / station order from the route file."""
    print("Loading delay file...")
    delays = pd.read_csv(DELAY_FILE)

    print("Loading route file...")
    routes = pd.read_csv(ROUTE_FILE)

    # Use the same column names everywhere
    delays["train_no"] = delays["train"].astype(str).str.strip()
    delays["station_code"] = delays["station"].astype(str).str.strip().str.upper()
    delays["date"] = pd.to_datetime(delays["date"], errors="coerce")
    delays["delay"] = pd.to_numeric(delays["arr_delay"], errors="coerce")
    delays["sched_hour"] = delays["sch_arr"].apply(time_text_to_hour)

    routes["train_no"] = routes["trainNumber"].astype(str).str.strip()
    routes["station_code"] = routes["station_code"].astype(str).str.strip().str.upper()
    routes["distance"] = pd.to_numeric(routes["distance"], errors="coerce")
    routes["route_idx"] = pd.to_numeric(routes["stnSerialNumber"], errors="coerce")
    routes["train_name"] = routes["trainName"].astype(str)

    # One row per train + station (used to add km and stop number)
    route_info = routes[
        ["train_no", "station_code", "distance", "route_idx", "train_name"]
    ].drop_duplicates(["train_no", "station_code"])

    df = delays.merge(route_info, on=["train_no", "station_code"], how="left")

    # Optional extra info: which railway zone the station belongs to
    try:
        with open(ZONE_FILE, "r", encoding="utf-8") as zone_file:
            zone_map = json.load(zone_file)
        df["zone"] = df["station_code"].map(zone_map)
    except FileNotFoundError:
        df["zone"] = None

    # Keep rows that have the numbers we need
    df = df.dropna(subset=["date", "delay", "distance", "sched_hour", "route_idx"])

    # Drop crazy delays (more than 24 hours)
    df = df[(df["delay"] >= 0) & (df["delay"] < 1440)]

    df = df.sort_values(["train_no", "date", "route_idx"]).reset_index(drop=True)

    print("Rows after cleaning:", len(df))
    print("Trains:", df["train_no"].nunique())
    return df


def split_by_date(df):
    """Past days = train set. Last few days = test set."""
    last_date = df["date"].max()
    cutoff_date = last_date - pd.Timedelta(days=TEST_DAYS)
    train = df[df["date"] <= cutoff_date].copy()
    test = df[df["date"] > cutoff_date].copy()
    print("Train rows:", len(train), "| Test rows:", len(test))
    print("Train dates up to:", cutoff_date.date(), "| Test after that")
    return train, test


def add_history_features(train, test):
    """
    avg_here = how late this train usually is at this station.
    We only use training days, so the test days stay unseen.
    """
    hist = train.groupby(["train_no", "station_code"])["delay"].agg(
        h_sum="sum", h_n="count"
    ).reset_index()

    station_avg = train.groupby("station_code")["delay"].mean().reset_index()
    station_avg = station_avg.rename(columns={"delay": "station_mean"})
    global_mean = train["delay"].mean()

    train = train.merge(hist, on=["train_no", "station_code"], how="left")
    train = train.merge(station_avg, on="station_code", how="left")
    test = test.merge(hist, on=["train_no", "station_code"], how="left")
    test = test.merge(station_avg, on="station_code", how="left")

    train["station_mean"] = train["station_mean"].fillna(global_mean)
    test["station_mean"] = test["station_mean"].fillna(global_mean)

    # For training rows, remove this row's own delay from the average.
    # Otherwise the model would "cheat" by seeing the answer.
    train["avg_here"] = (train["h_sum"] - train["delay"]) / (train["h_n"] - 1)
    train.loc[train["h_n"] <= 1, "avg_here"] = None
    train["avg_here"] = train["avg_here"].fillna(train["station_mean"])
    train["hist_n"] = train["h_n"] - 1

    test["avg_here"] = test["h_sum"] / test["h_n"]
    test["avg_here"] = test["avg_here"].fillna(test["station_mean"])
    test["hist_n"] = test["h_n"].fillna(0)

    return train, test, hist, station_avg, global_mean


def mean_abs_error(real_values, predicted_values):
    """Average of |predicted - real| in minutes. Lower is better."""
    return (predicted_values - real_values).abs().mean()


def make_model():
    """
    XGBoost regression model for train delay prediction.
    """
    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=120,
        learning_rate=0.08,
        max_depth=6,
        random_state=0,
        n_jobs=-1
    )


def train_mode_a(train, test):
    """Mode A: the train has not started yet. We only know the timetable."""
    model_a = make_model()
    model_a.fit(train[FEATURES_A], train["delay"])

    pred_a = model_a.predict(test[FEATURES_A])
    mae_baseline = mean_abs_error(test["delay"], test["avg_here"])
    mae_model = mean_abs_error(test["delay"], pred_a)
    return model_a, mae_baseline, mae_model


def make_pairs(data, max_rows):
    """
    Each row becomes: destination station + an earlier station on the same trip.
    This is how we teach Mode B (train already running).
    """
    keep = [
        "train_no", "date", "distance", "route_idx", "sched_hour",
        "delay", "avg_here", "hist_n", "station_mean",
    ]
    data = data[keep].copy()
    data = data.sort_values(["train_no", "date", "route_idx"])

    all_parts = []
    for k in LOOK_BACK_STOPS:
        part = data.copy()
        grouped = part.groupby(["train_no", "date"])
        part["prev_delay"] = grouped["delay"].shift(k)
        part["prev_avg"] = grouped["avg_here"].shift(k)
        part["prev_distance"] = grouped["distance"].shift(k)
        part["prev_route_idx"] = grouped["route_idx"].shift(k)
        part = part.dropna(subset=["prev_delay"])
        all_parts.append(part)

    pairs = pd.concat(all_parts, ignore_index=True)
    pairs["gap_km"] = pairs["distance"] - pairs["prev_distance"]
    pairs["stops_between"] = pairs["route_idx"] - pairs["prev_route_idx"]
    pairs["prev_excess"] = pairs["prev_delay"] - pairs["prev_avg"]

    # Keep a smaller sample so training stays reasonably fast
    if len(pairs) > max_rows:
        pairs = pairs.sample(max_rows, random_state=0)

    return pairs


def train_mode_b(train, test):
    """Mode B: we know the delay at an earlier station."""
    train_pairs = make_pairs(train, 400000)
    test_pairs = make_pairs(test, 100000)

    model_b = make_model()
    model_b.fit(train_pairs[FEATURES_B], train_pairs["delay"])
    pred_b = model_b.predict(test_pairs[FEATURES_B])

    # Simple rule with no machine learning:
    # delay at destination ~= current delay * a factor (by distance left)
    gap_ranges = [(0, 50), (50, 150), (150, 400), (400, 100000)]
    factors = []
    carry_pred = pd.Series(0.0, index=test_pairs.index)

    for low, high in gap_ranges:
        rows_train = train_pairs[
            (train_pairs["gap_km"] >= low) & (train_pairs["gap_km"] < high)
        ]
        if len(rows_train) > 0:
            top = (rows_train["prev_delay"] * rows_train["delay"]).sum()
            bottom = (rows_train["prev_delay"] ** 2).sum()
            if bottom == 0:
                factor = 1.0
            else:
                factor = top / bottom
        else:
            factor = 1.0
        factors.append((low, high, factor))

        in_range = (test_pairs["gap_km"] >= low) & (test_pairs["gap_km"] < high)
        carry_pred[in_range] = test_pairs.loc[in_range, "prev_delay"] * factor

    mae_baseline = mean_abs_error(test_pairs["delay"], test_pairs["avg_here"])
    mae_carry = mean_abs_error(test_pairs["delay"], carry_pred)
    mae_model = mean_abs_error(test_pairs["delay"], pred_b)
    return model_b, mae_baseline, mae_carry, mae_model, factors


def build_route_table(df):
    """One list of stations for each train (used by the app)."""
    routes = df.groupby(["train_no", "station_code"]).agg(
        distance=("distance", "median"),
        sched_hour=("sched_hour", "median"),
        route_idx=("route_idx", "median"),
    ).reset_index()
    routes = routes.sort_values(["train_no", "distance"])

    names = {}
    if "train_name" in df.columns:
        first_rows = df.drop_duplicates("train_no")
        names = dict(zip(first_rows["train_no"], first_rows["train_name"]))
    return routes, names


def main():
    df = load_and_clean()
    train, test = split_by_date(df)
    train, test, hist, station_avg, global_mean = add_history_features(train, test)

    print("\nTraining Mode A (train not started)...")
    model_a, mae_base_a, mae_model_a = train_mode_a(train, test)

    print("Training Mode B (train already running)...")
    model_b, mae_base_b, mae_carry_b, mae_model_b, factors = train_mode_b(train, test)

    print("\n====== RESULTS: average error in minutes (lower = better) ======")
    print("Train NOT started:")
    print("   Baseline (usual delay at station) :", round(mae_base_a, 1))
    print("   Tree model                        :", round(mae_model_a, 1))
    print("Train ALREADY running:")
    print("   Baseline (usual delay at station) :", round(mae_base_b, 1))
    print("   Carry-over rule (simple math)     :", round(mae_carry_b, 1))
    print("   Tree model                        :", round(mae_model_b, 1))

    print("\nCarry-over factors:")
    for low, high, factor in factors:
        print("  gap", low, "to", high, "km -> multiply current delay by", round(factor, 2))

    routes, names = build_route_table(df)
    joblib.dump(
        {
            "model_a": model_a,
            "model_b": model_b,
            "hist": hist,
            "station_mean": station_avg,
            "gmean": global_mean,
            "routes": routes,
            "names": names,
            "feats_a": FEATURES_A,
            "feats_b": FEATURES_B,
            "mae_a": mae_model_a,
            "mae_b": mae_model_b,
        },
        "artifacts.joblib",
    )
    print("\nSaved artifacts.joblib")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()

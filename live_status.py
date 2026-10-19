# =====================================================================
# LIVE STATUS  (RailRadar)
# The app calls get_live_status(train_no, route).
# It returns a dictionary with last_station and delay, or None.
# =====================================================================

import os
import requests
import streamlit as st

# "railradar" = real live data
# "demo"      = fake data for testing without an API key
MODE = "railradar"

RAILRADAR_URL = "https://api.railradar.in/v1/trains/{train_no}/live"
KEY_FILE = "railradar_key.txt"


def read_api_key():
    try:
        key = str(st.secrets["RAILRADAR_API_KEY"]).strip()
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass

    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            file_key = f.read().strip()
        if file_key and not file_key.startswith("PASTE"):
            return file_key

    return ""


def get_live_status(train_no, route):
    """route is the station table for this train."""
    if MODE == "demo":
        return demo_status(route)
    return railradar_status(train_no, route)


def demo_status(route):
    """Pretend the train is about 40% of the way, 35 minutes late."""
    middle_index = int(len(route) * 0.4)
    middle_station = route["station_code"].iloc[middle_index]
    return {
        "last_station": middle_station,
        "delay": 35,
        "status": "running",
        "train_name": "",
        "next_station": "",
        "message": "DEMO data (not real)",
        "ok_for_prediction": True,
    }


def pad_train_no(train_no):
    """RailRadar wants a 5-digit train number, like 01203 or 12303."""
    text = str(train_no).strip()
    if text.isdigit() and len(text) < 5:
        text = text.zfill(5)
    return text


def last_station_on_our_route(payload, route_codes):
    """Pick the latest live halt that also exists on our saved route."""
    live_route = payload.get("route") or []
    last_match = None
    for stop in live_route:
        status = str(stop.get("status") or "").lower()
        code = str(stop.get("stationCode") or "").upper()
        useful = status in ("departed", "arrived", "at-station", "halted")
        if useful and code in route_codes:
            last_match = code
    if last_match:
        return last_match

    current = payload.get("currentLocation") or {}
    current_code = str(current.get("stationCode") or "").upper()
    if current_code in route_codes:
        return current_code

    previous = payload.get("previousHalt") or {}
    previous_code = str(previous.get("stationCode") or "").upper()
    if previous_code in route_codes:
        return previous_code
    return None


def delay_from_payload(payload):
    """Use the overall delay, or the last known station delay."""
    overall = payload.get("delayMinutes")
    if overall is not None:
        try:
            return int(overall)
        except (TypeError, ValueError):
            pass

    live_route = payload.get("route") or []
    last_delay = 0
    for stop in live_route:
        arr = stop.get("delayArrival")
        dep = stop.get("delayDeparture")
        if arr is not None:
            last_delay = int(arr)
        elif dep is not None:
            last_delay = int(dep)
    return last_delay


def railradar_status(train_no, route):
    api_key = read_api_key()
    if api_key == "":
        return {
            "last_station": None,
            "delay": 0,
            "status": "no_key",
            "train_name": "",
            "next_station": "",
            "message": "No RailRadar API key found in railradar_key.txt",
            "ok_for_prediction": False,
        }

    url = RAILRADAR_URL.format(train_no=pad_train_no(train_no))
    try:
        reply = requests.get(
            url,
            headers={"Authorization": "Bearer " + api_key},
            timeout=15,
        )
    except Exception as error:
        return {
            "last_station": None,
            "delay": 0,
            "status": "error",
            "train_name": "",
            "next_station": "",
            "message": "Could not reach RailRadar: " + str(error),
            "ok_for_prediction": False,
        }

    if reply.status_code == 401:
        return {
            "last_station": None,
            "delay": 0,
            "status": "error",
            "train_name": "",
            "next_station": "",
            "message": "RailRadar rejected the API key (401). Check the key.",
            "ok_for_prediction": False,
        }
    if reply.status_code == 404:
        return {
            "last_station": None,
            "delay": 0,
            "status": "not_found",
            "train_name": "",
            "next_station": "",
            "message": "RailRadar has no live run for this train right now.",
            "ok_for_prediction": False,
        }
    if reply.status_code == 429:
        return {
            "last_station": None,
            "delay": 0,
            "status": "error",
            "train_name": "",
            "next_station": "",
            "message": "RailRadar rate limit reached. Try again later.",
            "ok_for_prediction": False,
        }
    if reply.status_code != 200:
        return {
            "last_station": None,
            "delay": 0,
            "status": "error",
            "train_name": "",
            "next_station": "",
            "message": "RailRadar HTTP " + str(reply.status_code),
            "ok_for_prediction": False,
        }

    body = reply.json()
    payload = body.get("data") or {}
    status = str(payload.get("status") or "").lower()
    train_name = str((payload.get("train") or {}).get("name") or payload.get("trainName") or "")
    next_halt = payload.get("nextHalt") or {}
    next_station = str(next_halt.get("stationCode") or "")

    route_codes = list(route["station_code"].astype(str).str.upper())
    last_station = last_station_on_our_route(payload, route_codes)
    delay = delay_from_payload(payload)
    station_times = [
        {
            "station_code": str(stop.get("stationCode") or "").upper(),
            "scheduled_arrival": stop.get("scheduledArrival"),
            "actual_arrival": stop.get("actualArrival"),
        }
        for stop in (payload.get("route") or [])
    ]

    # These statuses mean we should NOT use live delay for Mode B
    not_running = status in (
        "not_started", "scheduled", "yet_to_start", "upcoming",
        "cancelled", "canceled", "not found",
    )
    ok = (not not_running) and (last_station is not None)

    message = "Live status: " + (status or "unknown")
    if payload.get("isLive") is False:
        message = message + " (may be cached / not live)"
    if last_station is None and not not_running:
        message = message + ". Live station is not on this dataset route."

    return {
        "last_station": last_station,
        "delay": delay,
        "status": status or "unknown",
        "train_name": train_name,
        "next_station": next_station,
        "message": message,
        "ok_for_prediction": ok,
        "station_times": station_times,
    }

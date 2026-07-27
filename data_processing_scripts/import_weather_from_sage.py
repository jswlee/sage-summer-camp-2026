"""Fetch weather variables (temp, humidity, pressure) from Sage and merge into all_data.

For each node this queries the WXT weather sensor (wxt.env.temp/humidity/pressure).
If a node has no WXT data, it falls back to the AQT sensor (aqt.env.*).

Outputs:
  - weather_data.csv                  intermediate: timestamp, vsn, temp, humidity, pressure
  - all_data_with_weathervar.csv      all_data.csv + temp/humidity/pressure aligned by nearest timestamp
"""

from pathlib import Path

import pandas as pd
import sage_data_client

NODES = ["W0A4", "W09E", "W095", "W0A0", "W099"]
ALL_DATA_PATH = Path("all_data.csv")
WEATHER_PATH = Path("weather_data.csv")
OUTPUT_PATH = Path("all_data_with_weathervar.csv")

WEATHER_VARS = ["temp", "humidity", "pressure"]
SENSOR_PREFERENCE = ["wxt.env", "aqt.env"]
QUERY_BUFFER = pd.Timedelta("1d")


def fetch_weather(vsn: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    """Fetch temp/humidity/pressure for a node from a single sensor prefix.

    Returns a wide dataframe: timestamp, vsn, temp, humidity, pressure.
    Empty if the sensor produced no data for this node/window.
    """
    name_filter = "|".join(f"{prefix}.{var}" for var in WEATHER_VARS)
    df = sage_data_client.query(
        start=start,
        end=end,
        filter={"vsn": vsn, "name": name_filter},
    )
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df["var"] = df["name"].str.rsplit(".", n=1).str[1]

    wide = (
        df.pivot_table(index="timestamp", columns="var", values="value", aggfunc="mean")
        .reset_index()
        .sort_values("timestamp")
    )
    for var in WEATHER_VARS:
        if var not in wide.columns:
            wide[var] = pd.NA
    wide["vsn"] = vsn
    return wide[["timestamp", "vsn", *WEATHER_VARS]]


def fetch_node_weather(vsn: str, start: str, end: str) -> pd.DataFrame:
    """Fetch weather for a node, preferring WXT and falling back to AQT."""
    for prefix in SENSOR_PREFERENCE:
        wide = fetch_weather(vsn, prefix, start, end)
        if not wide.empty:
            print(f"  {vsn}: using {prefix}.* ({len(wide)} rows)")
            return wide
        print(f"  {vsn}: no data from {prefix}.*")
    print(f"  {vsn}: no weather data found")
    return pd.DataFrame(columns=["timestamp", "vsn", *WEATHER_VARS])


def build_weather_data(all_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather for all nodes over the span of all_data and save intermediate file."""
    start = (all_data["timestamp"].min() - QUERY_BUFFER).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (all_data["timestamp"].max() + QUERY_BUFFER).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Querying weather from {start} to {end}")

    frames = []
    for vsn in NODES:
        print(f"Processing node {vsn}...")
        frames.append(fetch_node_weather(vsn, start, end))

    weather = pd.concat(frames, ignore_index=True)
    weather.to_csv(WEATHER_PATH, index=False)
    print(f"Saved intermediate weather data to {WEATHER_PATH} ({len(weather)} rows)")
    return weather


def merge_weather(all_data: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Align nearest weather reading to each all_data row, per node."""
    for var in WEATHER_VARS:
        all_data[var] = pd.NA

    weather["timestamp"] = pd.to_datetime(weather["timestamp"]).dt.tz_convert("UTC")

    for vsn in all_data["vsn"].unique():
        node_mask = all_data["vsn"] == vsn
        node_rows = all_data.loc[node_mask].sort_values("timestamp")
        node_weather = weather.loc[weather["vsn"] == vsn].sort_values("timestamp")
        if node_weather.empty:
            print(f"  {vsn}: no weather to merge")
            continue

        merged = pd.merge_asof(
            node_rows[["timestamp"]],
            node_weather[["timestamp", *WEATHER_VARS]],
            on="timestamp",
            direction="nearest",
        )
        merged.index = node_rows.index
        for var in WEATHER_VARS:
            all_data.loc[node_rows.index, var] = merged[var]
        print(f"  {vsn}: merged weather into {len(node_rows)} rows")

    return all_data


def main() -> None:
    all_data = pd.read_csv(ALL_DATA_PATH)
    all_data["timestamp"] = pd.to_datetime(all_data["timestamp"]).dt.tz_convert("UTC")

    weather = build_weather_data(all_data)
    all_data = merge_weather(all_data, weather)

    all_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({len(all_data)} rows)")


if __name__ == "__main__":
    main()

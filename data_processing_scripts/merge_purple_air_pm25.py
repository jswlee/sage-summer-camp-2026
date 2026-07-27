"""Add a purple_air_pm25 column to all_data.csv using reference PurpleAir PM2.5 data.

For each node, the script reads the corresponding reference CSV from node_ref_pm/,
computes the median of all measurement columns (ignoring the provided "Average"),
matches each image row to the nearest reference timestamp, and keeps the value only
if the match is within 1 minute.

Reference DateTime values are assumed to be in America/Chicago local time and are
converted to UTC before matching against all_data.csv timestamps.
"""

from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

REF_DIR = Path("purple_air_ref_pm")
ALL_DATA_PATH = Path("all_data.csv")
REF_TIMESTAMP_COLUMN = "DateTime"
REF_AVERAGE_COLUMN = "Average"
REF_TIMEZONE = "America/Chicago"
MATCH_THRESHOLD = pd.Timedelta("1min")


def load_reference_pm25(node: str) -> pd.DataFrame:
    """Load and preprocess the PurpleAir reference data for a node."""
    ref_path = REF_DIR / f"{node}_ref_pm.csv"
    df = pd.read_csv(ref_path)

    measurement_cols = [
        col for col in df.columns
        if col not in (REF_TIMESTAMP_COLUMN, REF_AVERAGE_COLUMN)
    ]
    df["pm25_median"] = df[measurement_cols].median(axis=1, skipna=True)

    local_times = pd.to_datetime(df[REF_TIMESTAMP_COLUMN])
    df["timestamp"] = (
        local_times.dt.tz_localize(REF_TIMEZONE)
        .dt.tz_convert("UTC")
        .dt.as_unit("ns")
    )

    return df[["timestamp", "pm25_median"]].sort_values("timestamp")


def match_node_purple_air(node_data: pd.DataFrame, ref_data: pd.DataFrame) -> pd.Series:
    """Merge image rows with nearest PurpleAir PM2.5 and apply the 1-minute threshold."""
    ref_data = ref_data.rename(columns={"timestamp": "ref_timestamp"})
    sorted_node = node_data.sort_values("timestamp")
    merged = pd.merge_asof(
        sorted_node,
        ref_data,
        left_on="timestamp",
        right_on="ref_timestamp",
        direction="nearest",
    )
    time_diff = (merged["timestamp"] - merged["ref_timestamp"]).abs()
    values = merged["pm25_median"].where(time_diff <= MATCH_THRESHOLD)
    values.index = sorted_node.index
    return values.sort_index()


def main() -> None:
    all_data = pd.read_csv(ALL_DATA_PATH)
    all_data["timestamp"] = pd.to_datetime(all_data["timestamp"]).dt.tz_convert("UTC").dt.as_unit("ns")

    purple_air_values = pd.Series(index=all_data.index, dtype="float64")

    for node in all_data["vsn"].unique():
        print(f"Processing PurpleAir reference data for {node}...")
        node_mask = all_data["vsn"] == node
        node_data = all_data.loc[node_mask, ["timestamp"]].copy()
        ref_data = load_reference_pm25(node)
        node_values = match_node_purple_air(node_data, ref_data)
        print(f"  {node_values.notna().sum()} of {len(node_values)} rows matched")
        purple_air_values.loc[node_mask] = node_values

    all_data["purple_air_pm25"] = purple_air_values
    all_data.to_csv(ALL_DATA_PATH, index=False)
    matched_count = all_data["purple_air_pm25"].notna().sum()
    print(f"Saved {ALL_DATA_PATH} with {matched_count} matched PurpleAir PM2.5 values.")


if __name__ == "__main__":
    main()

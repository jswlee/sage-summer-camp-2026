import os
import subprocess
from pathlib import Path

import pandas as pd
import sage_data_client

NODES = ["W0A4", "W09E", "W095", "W0A0", "W099"]
START = "-14d"
IMAGE_TASK_FILTER = "imagesampler-.*"
PM25_NAME = "aqt.particle.pm2.5"


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if not already set."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def fetch_image_metadata(vsn: str) -> pd.DataFrame:
    """Fetch image upload URLs for a node."""
    df = sage_data_client.query(
        start=START,
        filter={
            "name": "upload",
            "vsn": vsn,
            "task": IMAGE_TASK_FILTER,
        },
    )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df["vsn"] = vsn
    df = df.rename(columns={"value": "url"})
    return df[["timestamp", "vsn", "url"]]


def fetch_pm25(vsn: str) -> pd.DataFrame:
    """Fetch PM2.5 readings for a node and save the raw results to a CSV."""
    df = sage_data_client.query(
        start=START,
        filter={
            "vsn": vsn,
            "name": PM25_NAME,
        },
    )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df["vsn"] = vsn
    df = df.rename(columns={"value": "pm2.5"})
    raw_dir = Path("sage_data")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{vsn}_pm25.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw PM2.5 data for {vsn} to {raw_path}")
    return df[["timestamp", "vsn", "pm2.5"]]


def match_node_data(vsn: str) -> pd.DataFrame:
    """Match each image URL to the nearest PM2.5 reading for that node."""
    images = fetch_image_metadata(vsn).sort_values("timestamp")
    pm25 = fetch_pm25(vsn).sort_values("timestamp")

    matched = pd.merge_asof(
        images,
        pm25,
        on="timestamp",
        by="vsn",
        direction="nearest",
    )

    url_parts = matched["url"].str.rsplit("/", n=1)
    matched["base_url"] = url_parts.str[0]
    matched["filename"] = url_parts.str[1]

    return matched[["timestamp", "vsn", "url", "base_url", "filename", "pm2.5"]]


def run_wget_for_node(vsn: str, username: str, password: str) -> None:
    """Run wget for a node's URL list into images/<vsn>/."""
    url_file = Path("sage_data") / f"{vsn}_urls.txt"
    if not url_file.exists():
        print(f"No URL file found for {vsn}, skipping wget.")
        return

    output_dir = Path("images") / vsn
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "wget",
        f"--user={username}",
        f"--password={password}",
        "-r",
        "-nc",
        "-P", str(output_dir),
        "-i", url_file,
    ]
    print(f"Running wget for {vsn}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"Warning: wget for {vsn} exited with status {result.returncode} "
            "(often caused by a few failed URL responses). Already-downloaded files are skipped."
        )


def main() -> None:
    load_env_file()

    username = os.environ.get("SAGE_USERNAME")
    password = os.environ.get("SAGE_PASSWORD")
    if not username or not password:
        raise RuntimeError("SAGE_USERNAME and SAGE_PASSWORD must be set (via .env or environment)")

    all_node_frames = []
    for vsn in NODES:
        print(f"Processing node {vsn}...")
        node_data = match_node_data(vsn)
        all_node_frames.append(node_data)
        print(f"  Found {len(node_data)} image/PM2.5 pairs for {vsn}")

    all_data = pd.concat(all_node_frames, ignore_index=True)
    all_data.to_csv("all_data.csv", index=False)
    print(f"Saved combined data to all_data.csv ({len(all_data)} rows)")

    url_dir = Path("sage_data")
    url_dir.mkdir(parents=True, exist_ok=True)
    for vsn in NODES:
        node_urls = all_data.loc[all_data["vsn"] == vsn, "url"]
        url_file = url_dir / f"{vsn}_urls.txt"
        node_urls.to_csv(url_file, index=False, header=False)
        print(f"Saved {len(node_urls)} URLs for {vsn} to {url_file}")

    for vsn in NODES:
        run_wget_for_node(vsn, username, password)


if __name__ == "__main__":
    main()
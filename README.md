# sage-summer-camp-2026

This repository contains data and code for the SAGE Summer Camp 2026 project, which explores the relationship between outdoor images from SAGE nodes and local air quality (PM2.5).

## Data collection

### Time window

All data was collected for the **14 days prior to 2026-07-24 17:00 UTC**, going back to 2026-07-10. This window was chosen because Chicago experienced historically bad air pollution on **July 16, 2026**. The 14-day window was also the maximum historical range available for direct data download from PurpleAir, because the API was not working reliably at the time of collection.

### SAGE node selection

SAGE nodes were selected using a single practical criterion: they needed to have image data for the previous two weeks, and we needed access to them. We were only granted access to the **NIREM** nodes. The five nodes used in this project are:

- `W0A4`
- `W09E`
- `W095`
- `W0A0`
- `W099`

### SAGE data

SAGE data was queried using `sage-data-client` (see `data_processing_scripts/import_data_from_sage.py`):

- **Images**: `upload` events from tasks matching `imagesampler-.*`
- **PM2.5**: `aqt.particle.pm2.5` values from each node's onboard air-quality sensor

For each image, the nearest in-node PM2.5 reading was matched by timestamp using `pd.merge_asof`. Raw SAGE PM2.5 series and image URL lists are kept in `sage_data_raw/`.

### Why SAGE PM2.5 was not used as ground truth

We originally intended to use the SAGE nodes' own `pm2.5` readings, but after downloading them the data was unusable for all nodes except `W0A4`: the series were either a flat line, physically unrealistic (reporting healthy air during Chicago's worst pollution of the window), or heavily fragmented. To keep PM2.5 values consistent across every node, we instead adopted a single unified ground-truth source, **PurpleAir**. The SAGE `pm2.5` column is still retained in `all_data.csv` for reference.

### PurpleAir ground truth

For each SAGE node we identified the geographically closest PurpleAir station and downloaded the past 14 days of measurements. Raw reference CSVs live in `purple_air_ref_pm/` (one per node, e.g. `W0A4_ref_pm.csv`). We downloaded specifically the US EPA PM2.5 (AQI) data, as we could use the EPA's air quality categories, where any value over 150 qualifies as "Unhealthy" (for all groups).

The merge step (`data_processing_scripts/merge_purple_air_pm25.py`) does the following:

- Each PurpleAir station reports several channels; the **median** across all measurement columns (ignoring the provided `Average`) is taken per timestamp to reject outliers.
- Reference timestamps are treated as `America/Chicago` local time and converted to UTC before matching.
- Each image row is matched to the nearest reference timestamp via `pd.merge_asof`, keeping the value only if the match is within **1 minute**.

The result is written back to `all_data.csv`, which contains one row per image:

- `timestamp` — UTC timestamp of the image
- `vsn` — SAGE node ID (also serves as the node identifier throughout the project)
- `url` / `base_url` / `filename` — image storage URL and filename
- `pm2.5` — SAGE node's own (mostly unusable) PM2.5 reading
- `purple_air_pm25` — median PurpleAir PM2.5 ground truth

## Dataset preparation

`data_processing_scripts/prepare_yolo_dataset.py` turns `all_data.csv` and the downloaded images into a YOLO classification dataset:

- **Labels**: images are labelled `bad` when `purple_air_pm25 >= 151`, otherwise `good`.
- **Day/night filter**: images can be restricted to daytime (05:00–21:00 Chicago time), nighttime, or both.
- **Per-day stratified split**: images are grouped by `(date, label)` and each group is split **70/20/10** into train/val/test. Splitting per day guarantees that no single day is assigned entirely to one split, so every day is represented across all three splits.
- **Class balancing**: after splitting, the majority class in each split is randomly downsampled so that every split has a **uniform 50/50 good/bad distribution**.

The committed datasets `yolo_dataset_daynight_224/` and `yolo_dataset_daynight_640/` are the `both` (day + night) variants at 224×224 and 640×640 resolution.

## Repository structure

```
.
├── all_data.csv                          # merged image + PM2.5 dataset
├── purple_air_ref_pm/                    # PurpleAir reference CSVs per node
├── sage_data_raw/                        # raw SAGE PM2.5 and image URL lists
├── data_processing_scripts/              # scripts to download and merge data
│   ├── import_data_from_sage.py
│   ├── merge_purple_air_pm25.py
│   └── prepare_yolo_dataset.py
├── training_scripts/                     # YOLO classification training
│   ├── train_yolo_classification.py
│   └── visualize_yolo_classification.py
├── yolo_dataset_daynight_224/            # 224×224 YOLO classification dataset
├── yolo_dataset_daynight_640/            # 640×640 YOLO classification dataset
├── model_training_and_inference_results/ # trained model artifacts
└── requirements.txt
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Collect SAGE data (requires `SAGE_USERNAME` and `SAGE_PASSWORD` in the environment or a `.env` file):

```bash
python data_processing_scripts/import_data_from_sage.py
```

Merge PurpleAir reference PM2.5 into `all_data.csv`:

```bash
python data_processing_scripts/merge_purple_air_pm25.py
```

Prepare a YOLO classification dataset:

```bash
python data_processing_scripts/prepare_yolo_dataset.py --time-of-day day --imgsz 224
```

Train a YOLO model:

```bash
python training_scripts/train_yolo_classification.py
```

## Model training and results

Training is handled by `training_scripts/train_yolo_classification.py` (with test-set inference and visualization in `training_scripts/visualize_yolo_classification.py`).

The best run, saved in `model_training_and_inference_results/`, used the following configuration:

- **Model**: `yolo26s-cls`
- **Batch size**: 8
- **Epochs**: 200
- **Patience**: 0 (no early stopping)
- **Image size**: 224×224 (trained on the `yolo_dataset_daynight_224` set)

This configuration reached a **top-1 accuracy of 0.94** on the test set. Because the dataset is balanced to a uniform 50/50 good/bad distribution, this is a meaningful accuracy rather than an artifact of class imbalance. It also demonstrates that a lightweight model running on small 224×224 inputs is enough to get strong results.

The full argument set for this run is stored in `model_training_and_inference_results/args.yaml`, alongside `results.csv`, training/validation plots, confusion matrices, and the trained `weights/`.

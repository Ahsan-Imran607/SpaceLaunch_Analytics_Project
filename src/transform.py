
import pandas as pd
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "launches_transformed.csv"
)

# ==========================================
# SELECT LATEST SUCCESSFUL EXTRACTION
# ==========================================

def get_latest_extraction():

    if not os.path.exists(RAW_FOLDER):
        raise FileNotFoundError("Raw data folder not found!")

    folders = sorted(
        [
            folder for folder in os.listdir(RAW_FOLDER)
            if folder.startswith("extraction_")
            and os.path.isdir(os.path.join(RAW_FOLDER, folder))
        ],
        reverse=True
    )

    for folder in folders:

        folder_path = os.path.join(RAW_FOLDER, folder)

        summary_path = os.path.join(
            folder_path,
            "summary.json"
        )

        if not os.path.exists(summary_path):
            continue

        try:

            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)

            if summary.get("status") == "completed":

                print("Selected extraction:", folder)

                return folder_path

        except (json.JSONDecodeError, OSError):
            continue

    raise FileNotFoundError(
        "No successful extraction found!"
    )


# ==========================================
# EXTRACT NESTED JSON VALUES
# ==========================================

def get_nested_value(data, *keys):

    for key in keys:

        if not isinstance(data, dict):
            return None

        data = data.get(key)

    return data


# ==========================================
# STANDARDIZE TEXT VALUES
# ==========================================

def clean_text(value):

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    value = value.strip()

    if value == "":
        return None

    return value


# ==========================================
# EXTRACT RAW JSON RECORDS
# ==========================================

def read_raw_data(folder_path):

    all_records = []

    files = sorted(
        file for file in os.listdir(folder_path)
        if file.startswith("page_")
        and file.endswith(".json")
    )

    if not files:
        raise FileNotFoundError(
            "No JSON page files found in extraction!"
        )

    for filename in files:

        file_path = os.path.join(
            folder_path,
            filename
        )

        print("Reading:", filename)

        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        records = raw_data.get("results", [])

        if not isinstance(records, list):
            raise ValueError(
                f"Invalid results in {filename}"
            )

        all_records.extend(records)

    print("\nTotal raw records:", len(all_records))

    return all_records


# ==========================================
# TRANSFORM INDIVIDUAL LAUNCH RECORD
# ==========================================

def transform_record(record):

    # Launch identification
    launch_id = record.get("id")

    launch_name = record.get("name")

    # Launch date and time
    launch_datetime = record.get("net")

    # Launch status
    status = get_nested_value(
        record,
        "status",
        "name"
    )

    # Launch service provider
    provider = get_nested_value(
        record,
        "launch_service_provider",
        "name"
    )

    # Rocket configuration
    rocket = get_nested_value(
        record,
        "rocket",
        "configuration",
        "name"
    )

    # Launch location
    pad = record.get("pad") or {}

    launch_location = pad.get("name")

    if not launch_location:
        launch_location = get_nested_value(
            pad,
            "location",
            "name"
        )

    # Country
    country = get_nested_value(
        pad,
        "location",
        "country",
        "name"
    )

    # Handle country returned as a string
    if not country:

        country = get_nested_value(
            pad,
            "location",
            "country"
        )

    # Create transformed record
    transformed_record = {

        "launch_id": clean_text(launch_id),

        "launch_name": clean_text(launch_name),

        "launch_datetime": clean_text(launch_datetime),

        "launch_year": None,

        "launch_month": None,

        "provider": clean_text(provider),

        "rocket": clean_text(rocket),

        "launch_location": clean_text(launch_location),

        "country": clean_text(country),

        "status": clean_text(status)

    }

    return transformed_record


# ==========================================
# TRANSFORMATION FUNCTION
# ==========================================

def transform_data():

    print("Starting Data Transformation...")
    print("------------------------------------------")

    # Find latest successful extraction
    extraction_folder = get_latest_extraction()

    # Read raw JSON files
    raw_records = read_raw_data(extraction_folder)

    # Transform all records
    transformed_records = []

    for record in raw_records:

        transformed_record = transform_record(record)

        transformed_records.append(transformed_record)

    # Create Pandas DataFrame
    df = pd.DataFrame(transformed_records)

    # ======================================
    # STANDARDIZE DATETIME
    # ======================================

    df["launch_datetime"] = pd.to_datetime(
        df["launch_datetime"],
        errors="coerce",
        utc=True
    )

    # Derive year and month
    df["launch_year"] = (
        df["launch_datetime"].dt.year.astype("Int64")
    )

    df["launch_month"] = (
        df["launch_datetime"].dt.month.astype("Int64")
    )

    # Convert datetime into ISO format
    df["launch_datetime"] = (
        df["launch_datetime"].apply(
            lambda value: value.isoformat()
            if pd.notna(value) else None
        )
    )

    # ======================================
    # CREATE PROCESSED FOLDER
    # ======================================

    os.makedirs(
        PROCESSED_FOLDER,
        exist_ok=True
    )

    # ======================================
    # EXPORT CSV
    # ======================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ======================================
    # DISPLAY SUMMARY
    # ======================================

    print("\n------------------------------------------")
    print("TRANSFORMATION COMPLETED")
    print("------------------------------------------")

    print("Total records:", len(df))

    print("Total columns:", len(df.columns))

    print("\nDataset columns:")

    for column in df.columns:
        print("-", column)

    print("\nCSV saved at:", OUTPUT_FILE)

    print("\nDataFrame preview:")

    print(df.head())

    print("\nMissing values:")

    print(df.isnull().sum())

    return df


# ==========================================
# RUN TRANSFORMATION
# ==========================================

if __name__ == "__main__":

    df = transform_data()

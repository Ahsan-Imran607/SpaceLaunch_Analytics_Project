
import requests
import json
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

API_URL = "https://ll.thespacedevs.com/2.3.0/launches/previous/"

RAW_FOLDER = "data/raw"

START_DATE = "2020-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

PAGE_SIZE = 100
MAX_RECORDS = 1000

# Free API limit: 15 requests per hour
REQUEST_DELAY = 1

# Maximum retries for failed requests
MAX_RETRIES = 3

# ==========================================
# CREATE TIMESTAMPED FOLDER
# ==========================================

extraction_time = datetime.now()

folder_name = "extraction_" + extraction_time.strftime(
    "%Y%m%d_%H%M%S"
)

EXTRACTION_FOLDER = os.path.join(
    RAW_FOLDER,
    folder_name
)

os.makedirs(EXTRACTION_FOLDER, exist_ok=True)

# ==========================================
# API PARAMETERS
# ==========================================

params = {
    "net__gte": START_DATE,
    "net__lt": END_DATE,
    "limit": PAGE_SIZE,
    "mode": "detailed",
    "format": "json"
}

# ==========================================
# EXTRACTION FUNCTION
# ==========================================

def extract_data():

    next_url = API_URL

    total_records = 0
    page_number = 1
    failed_requests = 0

    start_time = time.time()

    session = requests.Session()

    print("Starting Rocket Launch Data Extraction...")
    print("------------------------------------------")

    while next_url and total_records < MAX_RECORDS:

        try:

            print(f"\nFetching page {page_number}...")

            if page_number > 1:
                print(f"Waiting {REQUEST_DELAY} seconds...")
                time.sleep(REQUEST_DELAY)

            # ==================================
            # SEND API REQUEST WITH RETRIES
            # ==================================

            response = None

            for attempt in range(1, MAX_RETRIES + 1):

                try:

                    if page_number == 1:

                        response = session.get(
                            next_url,
                            params=params,
                            timeout=30
                        )

                    else:

                        response = session.get(
                            next_url,
                            timeout=30
                        )

                    # Handle rate limiting
                    if response.status_code == 429:

                        if attempt < MAX_RETRIES:

                            wait_time = 60 * attempt

                            print(
                                f"Rate limited. "
                                f"Retrying in {wait_time} seconds..."
                            )

                            time.sleep(wait_time)
                            continue

                        response.raise_for_status()

                    response.raise_for_status()

                    break

                except requests.exceptions.RequestException:

                    if attempt == MAX_RETRIES:
                        raise

                    wait_time = 5 * attempt

                    print(
                        f"Request failed. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

            # ==================================
            # GET RAW JSON RESPONSE
            # ==================================

            raw_data = response.json()

            records = raw_data.get("results", [])

            if not isinstance(records, list):
                raise ValueError("Invalid results format in API response")

            # ==================================
            # APPLY EXACT RECORD LIMIT
            # ==================================

            remaining_records = MAX_RECORDS - total_records

            selected_records = records[:remaining_records]

            # Preserve the API response structure
            saved_data = raw_data.copy()
            saved_data["results"] = selected_records

            # ==================================
            # SAVE ORIGINAL JSON RESPONSE
            # ==================================

            filename = f"page_{page_number:03d}.json"

            file_path = os.path.join(
                EXTRACTION_FOLDER,
                filename
            )

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(
                    raw_data,
                    file,
                    ensure_ascii=False,
                    indent=4
    )

            # If the API returned more records than
            # our limit, save only the selected records.
            if len(selected_records) < len(records):

                with open(file_path, "w", encoding="utf-8") as file:

                    json.dump(
                        saved_data,
                        file,
                        ensure_ascii=False,
                        indent=4
                    )

            records_count = len(selected_records)

            total_records += records_count

            print(f"Saved: {filename}")
            print(f"Records in page: {records_count}")
            print(f"Total records: {total_records}")

            # ==================================
            # GET NEXT PAGE
            # ==================================

            if total_records >= MAX_RECORDS:
                break

            next_url = raw_data.get("next")

            page_number += 1

        except requests.exceptions.RequestException as error:

            print("\nAPI request failed:", error)

            failed_requests += 1
            break

        except (ValueError, AttributeError, TypeError) as error:

            print("\nInvalid API response:", error)

            failed_requests += 1
            break

    session.close()

    # ==========================================
    # EXTRACTION SUMMARY
    # ==========================================

    end_time = time.time()

    total_time = round(end_time - start_time, 2)

    summary = {

        "extraction_date": extraction_time.isoformat(),

        "start_date": START_DATE,

        "end_date": END_DATE,

        "total_records": total_records,

        "total_pages": page_number,

        "failed_requests": failed_requests,

        "extraction_time_seconds": total_time,

        "status": (
            "completed"
            if not failed_requests and (
                total_records >= MAX_RECORDS or not next_url
            )
            else "incomplete"
        )

    }

    summary_path = os.path.join(
        EXTRACTION_FOLDER,
        "summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ==========================================
    # DISPLAY FINAL SUMMARY
    # ==========================================

    print("\n------------------------------------------")
    print("EXTRACTION SUMMARY")
    print("------------------------------------------")

    print("Total records:", total_records)
    print("Total pages:", page_number)
    print("Failed requests:", failed_requests)
    print("Time taken:", total_time, "seconds")
    print("Status:", summary["status"])

    print("\nRaw data location:")
    print(EXTRACTION_FOLDER)

    print("\nExtraction completed!")


# ==========================================
# RUN EXTRACTION
# ==========================================

if __name__ == "__main__":
    extract_data()

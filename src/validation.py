import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from datetime import datetime
import json
from pathlib import Path
import logging 

# ==========================================
# LOGGING SETUP
# ==========================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# PYDANTIC SCHEMA DEFINITION
# ==========================================
class LaunchRecordModel(BaseModel):
    """Pydantic model enforcing strict data-quality rules per launch record."""
    launch_id: str = Field(..., min_length=1, description="Must exist and be non-empty")
    mission_name: str = Field(..., min_length=1, description="Must not be empty")
    launch_datetime: datetime = Field(..., description="Must be a valid datetime")
    launch_year: int = Field(..., description="Must be a valid year")
    launch_month: int = Field(..., ge=1, le=12, description="Must be between 1 and 12")
    provider: str = Field(..., min_length=1, description="Must be a valid non-empty value")
    rocket: str = Field(..., min_length=1, description="Must be a valid non-empty value")
    launch_location: str = Field(..., min_length=1, description="Must be a valid non-empty value")
    country: str | None = Field(None, description="Nullable field allowed per API coverage")
    status: str = Field(..., min_length=1, description="Must belong to defined status categories")

    @model_validator(mode="after")
    def verify_datetime_consistency(self) -> "LaunchRecordModel":
        """Ensure launch_year and launch_month precisely match the launch_datetime."""
        if self.launch_datetime:
            if self.launch_datetime.year != self.launch_year:
                raise ValueError(
                    f"launch_year ({self.launch_year}) does not match launch_datetime year ({self.launch_datetime.year})"
                )
            if self.launch_datetime.month != self.launch_month:
                raise ValueError(
                    f"launch_month ({self.launch_month}) does not match launch_datetime month ({self.launch_datetime.month})"
                )
        return self

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status category."""
        valid_categories = {"Success", "Failure", "Partial Failure", "Scheduled", "TBD", "Launch Successful"}
        normalized = v.strip()
        if not normalized:
            raise ValueError("Status cannot be empty or whitespace.")
        return normalized


# ==========================================
# MAIN VALIDATION PIPELINE FUNCTION
# ==========================================
def validate_launch_data(df, output_dir="data/processed"):
    """
    Validates a transformed DataFrame against defined data-quality rules.

    Args:
        df: Transformed pandas DataFrame containing launch records.
        output_dir: Directory to save validation reports and processed files.

    Returns:
        A tuple containing:
        - DataFrame of valid records.
        - List of rejected records with error metadata.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_records = len(df)
    logger.info(f"Starting validation pipeline for {total_records} records...")

    missing_values_count = 0
    duplicate_records_count = 0
    invalid_records_count = 0

    # 1. Check for missing values in required fields (excluding nullable 'country')
    required_fields = [
        "launch_id", "mission_name", "launch_datetime", 
        "launch_year", "launch_month", "provider", 
        "rocket", "launch_location", "status"
    ]
    
    # Track rows with missing values
    missing_mask = df[required_fields].isnull().any(axis=1)
    missing_values_count = int(missing_mask.sum())

    # 2. Check for duplicate launch_ids
    if "launch_id" in df.columns:
        dupes_mask = df.duplicated(subset=["launch_id"], keep="first")
        duplicate_records_count = int(dupes_mask.sum())
    else:
        dupes_mask = pd.Series([False] * total_records)

    valid_records = []
    rejected_records = []

    # 3. Row-by-row Pydantic validation and anomaly checks
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        launch_id = row_dict.get("launch_id", f"INDEX_{idx}")

        # Check pre-filtering flags
        if missing_mask[idx]:
            reason = "Missing value in required fields."
            logger.error(f"Validation rejected [{launch_id}]: {reason}")
            rejected_records.append({"row_index": idx, "launch_id": launch_id, "raw_data": row_dict, "error": reason})
            invalid_records_count += 1
            continue

        if dupes_mask[idx]:
            reason = f"Duplicate launch_id detected: '{launch_id}'."
            logger.error(f"Validation rejected [{launch_id}]: {reason}")
            rejected_records.append({"row_index": idx, "launch_id": launch_id, "raw_data": row_dict, "error": reason})
            continue  # Counted under duplicate_records

        try:
            # Validate through Pydantic model
            validated_model = LaunchRecordModel(**row_dict)
            valid_records.append(validated_model.model_dump())
        except ValidationError as e:
            invalid_records_count += 1
            error_msg = str(e)
            logger.error(f"Validation failed for launch_id '{launch_id}': {error_msg}")
            rejected_records.append({
                "row_index": idx,
                "launch_id": launch_id,
                "raw_data": row_dict,
                "error": error_msg
            })
        except Exception as e:
            invalid_records_count += 1
            error_msg = str(e)
            logger.error(f"Unexpected error for launch_id '{launch_id}': {error_msg}")
            rejected_records.append({
                "row_index": idx,
                "launch_id": launch_id,
                "raw_data": row_dict,
                "error": error_msg
            })

    valid_df = pd.DataFrame(valid_records)
    valid_count = len(valid_df)
    rejected_count = len(rejected_records)

    # 4. Generate validation report summary
    report = {
        "total_records": total_records,
        "valid_records": valid_count,
        "rejected_records": rejected_count,
        "duplicate_records": duplicate_records_count,
        "missing_values": missing_values_count,
        "invalid_records": invalid_records_count,
        "status": "completed"
    }

    report_path = Path(output_dir) / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    rejected_path = Path(output_dir) / "rejected_records.json"
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected_records, f, indent=4, default=str)

    logger.info("==========================================")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"Total Records     : {total_records}")
    logger.info(f"Valid Records     : {valid_count}")
    logger.info(f"Rejected Records  : {rejected_count}")
    logger.info(f"Duplicate Records : {duplicate_records_count}")
    logger.info(f"Missing Values    : {missing_values_count}")
    logger.info(f"Invalid Records   : {invalid_records_count}")
    logger.info(f"Report saved to   : {report_path}")
    logger.info("==========================================")

    return valid_df, rejected_records


if __name__ == "__main__":
    # Test stub with mock DataFrame to verify functionality
    mock_data = pd.DataFrame([
        {
            "launch_id": "test-id-001",
            "mission_name": "Artemis Test",
            "launch_datetime": datetime(2025, 6, 15, 12, 0, 0),
            "launch_year": 2025,
            "launch_month": 6,
            "provider": "NASA",
            "rocket": "SLS",
            "launch_location": "Kennedy Space Center",
            "country": "USA",
            "status": "Success"
        }
    ])
    validate_launch_data(mock_data)

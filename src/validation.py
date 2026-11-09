"""
Validation stage for the rocket-launch pipeline.

Expects the flat DataFrame produced by transform.py
(data/processed/launches_transformed.csv). Use load_transformed_csv() to read it:
plain pd.read_csv() would turn a pad name like "201" into an int.

    launch_id, launch_name, launch_datetime (ISO, UTC), launch_year, launch_month,
    provider, rocket, launch_location, country (nullable), status

Optional column `last_updated` (not currently emitted by transform.py) is used to pick the
newest copy when the same launch_id appears more than once.

Every input row ends up in exactly one bucket:
    valid | rejected as "missing" | rejected as "invalid" | rejected as "duplicate"
so that  valid + missing + invalid + duplicate == total.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# ==========================================
# CONFIGURATION (keep in sync with extract.py)
# ==========================================
START_DATE = "2020-01-01T00:00:00Z"   # inclusive
END_DATE = "2026-01-01T00:00:00Z"     # exclusive (extract.py uses net__lt)

_START_DT = datetime.fromisoformat(START_DATE.replace("Z", "+00:00"))
_END_DT = datetime.fromisoformat(END_DATE.replace("Z", "+00:00"))

REQUIRED_FIELDS = [
    "launch_id", "launch_name", "launch_datetime",
    "launch_year", "launch_month", "provider",
    "rocket", "launch_location", "status",
]
OPTIONAL_FIELDS = ["country"]

# Paths used by transform.py / extract.py (relative to the project root)
RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"
TRANSFORMED_CSV = os.path.join(PROCESSED_FOLDER, "launches_transformed.csv")

# Launch Library 2 ids are UUIDs; set False if you ever feed non-UUID ids.
ENFORCE_UUID_IDS = True

# Canonical status names as returned by Launch Library 2 (status.name),
# plus a map from the short form (status.abbrev) so either can be used.
VALID_STATUSES = {
    "Go for Launch",
    "To Be Confirmed",
    "To Be Determined",
    "On Hold",
    "Launch in Flight",
    "Launch Successful",
    "Launch Failure",
    "Launch was a Partial Failure",
}
STATUS_ABBREV_TO_NAME = {
    "Go": "Go for Launch",
    "TBC": "To Be Confirmed",
    "TBD": "To Be Determined",
    "Hold": "On Hold",
    "In Flight": "Launch in Flight",
    "Success": "Launch Successful",
    "Failure": "Launch Failure",
    "Partial Failure": "Launch was a Partial Failure",
}
# Statuses that mean "the launch already happened" -> date must not be in the future.
FINAL_STATUSES = {"Launch Successful", "Launch Failure", "Launch was a Partial Failure"}
FUTURE_TOLERANCE = timedelta(days=1)  # tolerate clock skew / late status updates

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # importing this module has no side effects


class _ConsoleFormatter(logging.Formatter):
    """Plain text for INFO; 'WARNING: ...' / 'ERROR: ...' prefix for anything else."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        return msg if record.levelno == logging.INFO else f"{record.levelname}: {msg}"


def configure_logging(log_dir: str | Path = "logs") -> None:
    """
    Opt-in logging setup (call from your entry-point, not at import time).

    Console : clean summary only (no per-row error text).
    Log file: logs/validation.log, with timestamps and the full reason for every rejected row.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_dir / "validation.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console)


# ==========================================
# PYDANTIC SCHEMA
# ==========================================
class LaunchRecordModel(BaseModel):
    """Strict data-quality rules for one launch record."""

    model_config = ConfigDict(str_strip_whitespace=True, coerce_numbers_to_str=True)  # pad names like "201" survive CSV round-trips

    launch_id: str = Field(..., min_length=1)
    launch_name: str = Field(..., min_length=1)
    launch_datetime: datetime
    launch_year: int = Field(..., ge=_START_DT.year, le=_END_DT.year)
    launch_month: int = Field(..., ge=1, le=12)
    provider: str = Field(..., min_length=1)
    rocket: str = Field(..., min_length=1)
    launch_location: str = Field(..., min_length=1)
    country: str | None = None  # nullable per API coverage
    status: str = Field(..., min_length=1)

    # --- field-level validators -------------------------------------------
    @field_validator("launch_id")
    @classmethod
    def _launch_id_is_uuid(cls, v: str) -> str:
        if ENFORCE_UUID_IDS:
            try:
                # canonical lowercase, hyphenated form -> case/format variants of the
                # same UUID are recognised as duplicates
                return str(uuid.UUID(v))
            except ValueError:
                raise ValueError(f"launch_id '{v}' is not a valid UUID")
        return v

    @field_validator("launch_datetime", mode="before")
    @classmethod
    def _datetime_must_be_text_or_datetime(cls, v: Any) -> Any:
        # Pydantic's lax mode would turn ints/floats into 1970-era timestamps.
        if not isinstance(v, (str, datetime)):
            raise ValueError(f"launch_datetime must be an ISO string or datetime, got {type(v).__name__}")
        return v

    @field_validator("launch_datetime", mode="after")
    @classmethod
    def _force_utc(cls, v: datetime) -> datetime:
        # Naive datetimes are assumed to already be UTC (the API reports UTC).
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @field_validator("country", mode="before")
    @classmethod
    def _blank_country_is_none(cls, v: Any) -> Any:
        # NaN / "" / whitespace -> None, so nullable really means nullable.
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        # transform.py's country fallback can leak the raw dict/list when the API
        # gives a country object without a "name"; it lands in the CSV as "{'id': ...".
        if isinstance(v, (dict, list)) or (isinstance(v, str) and v.lstrip().startswith(("{", "["))):
            raise ValueError(f"country looks like a raw JSON object, not a name: {str(v)[:60]}")
        return v

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        canonical = STATUS_ABBREV_TO_NAME.get(v, v)
        if canonical not in VALID_STATUSES:
            raise ValueError(
                f"Unknown status '{v}'. Expected one of {sorted(VALID_STATUSES)} "
                f"or abbreviations {sorted(STATUS_ABBREV_TO_NAME)}."
            )
        return canonical  # store the canonical name

    # --- cross-field validators -------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> "LaunchRecordModel":
        dt = self.launch_datetime  # already UTC

        if dt.year != self.launch_year:
            raise ValueError(f"launch_year ({self.launch_year}) != launch_datetime year ({dt.year}, UTC)")
        if dt.month != self.launch_month:
            raise ValueError(f"launch_month ({self.launch_month}) != launch_datetime month ({dt.month}, UTC)")

        if not (_START_DT <= dt < _END_DT):
            raise ValueError(f"launch_datetime {dt.isoformat()} outside extraction window [{START_DATE}, {END_DATE})")

        if self.status in FINAL_STATUSES and dt > datetime.now(timezone.utc) + FUTURE_TOLERANCE:
            raise ValueError(f"status '{self.status}' but launch_datetime {dt.isoformat()} is in the future")
        return self


# ==========================================
# HELPERS
# ==========================================
def _is_blank(v: Any) -> bool:
    """True for None, NaN/NaT, and empty / whitespace-only strings."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_python(v: Any) -> Any:
    """Convert numpy / pandas scalars to plain Python; blanks -> None."""
    if _is_blank(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()  # numpy scalar -> python scalar
        except (ValueError, AttributeError):
            return v
    return v


def _json_safe(obj: Any) -> Any:
    """Make rejected-record payloads valid JSON (NaN -> null, datetimes -> ISO)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return obj


def _reject(rejected: list, idx: Any, row_number: int, launch_id: Any, row: dict, category: str, reason: str) -> None:
    logger.debug("Row %d [%s] rejected (%s): %s", row_number, launch_id, category, reason)  # log file only
    rejected.append(
        {
            "row_number": row_number,  # 1-based position among the data rows (header excluded)
            "row_index": idx,
            "launch_id": launch_id,
            "category": category,
            "error": reason,
            "raw_data": row,
        }
    )


def _format_summary(report: dict, rejected: list, paths: dict[str, str]) -> str:
    """Build the boxed console summary."""
    width = 64
    total = report["total_records"]
    pct = lambda n: f"{(n / total * 100):.1f}%" if total else "-"
    heavy, light = "=" * width, "-" * width

    lines = [
        "",
        heavy,
        f" VALIDATION RESULT{report['status'].upper().rjust(width - 18)}",
        heavy,
        f" Records checked      {total:>6}",
        f" Valid                {report['valid_records']:>6}   ({pct(report['valid_records'])})",
        f" Rejected             {report['rejected_records']:>6}   ({pct(report['rejected_records'])})",
        f"    Missing values    {report['missing_values']:>6}",
        f"    Invalid values    {report['invalid_records']:>6}",
        f"    Duplicates        {report['duplicate_records']:>6}",
    ]
    if report.get("count_matches_expected") is not None:
        ok = "yes" if report["count_matches_expected"] else f"NO (extraction saved {report['expected_records']})"
        lines.append(f" Matches extraction   {ok:>6}")

    if rejected:
        lines += [light, " Rows with problems (row 1 = first data row, header not counted)"]
        for cat, label in (("missing", "Missing"), ("invalid", "Invalid"), ("duplicate", "Duplicate")):
            rows = sorted(r["row_number"] for r in rejected if r["category"] == cat)
            if rows:
                head = f"   {label} ({len(rows)}):".ljust(17) + " "
                body = textwrap.fill(
                    ", ".join(map(str, rows)),
                    width=width,
                    initial_indent=head,
                    subsequent_indent=" " * len(head),
                )
                lines.append(body)
    else:
        lines += [light, " No problems found."]

    lines += [light]
    lines += [f" {name:<9}: {path}" for name, path in paths.items()]
    lines.append(heavy)
    return "\n".join(lines)


# ==========================================
# MAIN PIPELINE FUNCTION
# ==========================================
def validate_launch_data(
    df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
    expected_records: int | None = None,
    max_reject_rate: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Validate a transformed launch DataFrame.

    Args:
        df: Flat DataFrame, one row per launch (see module docstring).
        output_dir: Where validation_report.json, rejected_records.json and
            valid_launches.csv are written.
        expected_records: Optional number of records you expect (e.g. the
            `total_records` from extract.py's summary.json). A mismatch is
            logged and recorded in the report.
        max_reject_rate: Optional threshold (0-1). If the share of rejected
            rows exceeds it, report["status"] is "failed_threshold".

    Returns:
        (valid_df, rejected_records)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(df)
    logger.info("Validating %d records...", total)

    # -- schema-level checks (fail loudly, not with a KeyError mid-run) -------
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
    unexpected = [c for c in df.columns if c not in REQUIRED_FIELDS + OPTIONAL_FIELDS + ["last_updated"]]
    if unexpected:
        logger.warning("Ignoring unexpected columns: %s", unexpected)

    has_updated = "last_updated" in df.columns
    position = {idx: n + 1 for n, idx in enumerate(df.index)}  # 1-based data-row number
    rejected: list[dict] = []
    candidates: list[tuple[Any, dict, dict]] = []  # (idx, raw_row, validated_dict)

    # -- pass 1: missing values + pydantic validation, one bucket per row -----
    for idx, raw in zip(df.index, df.to_dict("records")):
        row = {k: _to_python(v) for k, v in raw.items()}
        launch_id = row.get("launch_id") or f"INDEX_{idx}"

        missing = [f for f in REQUIRED_FIELDS if row.get(f) is None]
        if missing:
            _reject(rejected, idx, position[idx], launch_id, row, "missing", f"Missing/blank required fields: {missing}")
            continue

        try:
            model = LaunchRecordModel(**{k: row.get(k) for k in REQUIRED_FIELDS + OPTIONAL_FIELDS})
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'record'}: {err['msg']}" for err in e.errors())
            _reject(rejected, idx, position[idx], launch_id, row, "invalid", msg)
            continue
        except Exception as e:  # defensive: never let one row kill the batch
            _reject(rejected, idx, position[idx], launch_id, row, "invalid", f"Unexpected error: {e}")
            continue

        candidates.append((idx, row, model.model_dump()))

    # -- pass 2: de-duplicate AFTER validation so a bad copy can't hide a good one
    by_id: dict[str, list[tuple[Any, dict, dict]]] = {}
    for item in candidates:
        by_id.setdefault(item[2]["launch_id"], []).append(item)

    order = {idx: n for n, idx in enumerate(df.index)}
    kept: list[tuple[int, dict]] = []
    for launch_id, group in by_id.items():
        if len(group) > 1 and has_updated:
            # keep the most recently updated copy (ISO strings sort correctly)
            group.sort(key=lambda g: str(g[1].get("last_updated") or ""), reverse=True)
        keeper, losers = group[0], group[1:]
        kept.append((order[keeper[0]], keeper[2]))
        for idx, row, _ in losers:
            _reject(rejected, idx, position[idx], launch_id, row, "duplicate", f"Duplicate launch_id '{launch_id}'")

    valid_records = [rec for _, rec in sorted(kept, key=lambda t: t[0])]  # original row order

    columns = REQUIRED_FIELDS + OPTIONAL_FIELDS
    valid_df = pd.DataFrame(valid_records, columns=columns)  # keeps schema even when empty

    # -- counters: each rejected row is in exactly one category ---------------
    by_cat = {c: sum(1 for r in rejected if r["category"] == c) for c in ("missing", "invalid", "duplicate")}
    valid_count, rejected_count = len(valid_df), len(rejected)
    assert valid_count + rejected_count == total, "row accounting error"

    reject_rate = rejected_count / total if total else 0.0
    status = "completed"
    if max_reject_rate is not None and reject_rate > max_reject_rate:
        status = "failed_threshold"

    report = {
        "total_records": total,
        "valid_records": valid_count,
        "rejected_records": rejected_count,
        "duplicate_records": by_cat["duplicate"],
        "missing_values": by_cat["missing"],
        "invalid_records": by_cat["invalid"],
        "reject_rate": round(reject_rate, 4),
        "expected_records": expected_records,
        "count_matches_expected": (total == expected_records) if expected_records is not None else None,
        "status": status,
    }
    if expected_records is not None and total != expected_records:
        logger.warning("Record count mismatch: got %d, expected %d", total, expected_records)

    # -- outputs ---------------------------------------------------------------
    report_path = out / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    (out / "rejected_records.json").write_text(
        json.dumps(_json_safe(rejected), indent=4, default=str, allow_nan=False), encoding="utf-8"
    )
    valid_df.to_csv(out / "valid_launches.csv", index=False)

    logger.info(
        _format_summary(
            report,
            rejected,
            {
                "Report": str(report_path),
                "Rejected": f"{out / 'rejected_records.json'}  (reason for each row)",
                "Valid": str(out / "valid_launches.csv"),
                "Log": "logs/validation.log" if logger.handlers and any(
                    isinstance(h, logging.FileHandler) for h in logger.handlers) else "(logging to file off)",
            },
        )
    )

    return valid_df, rejected


def get_expected_records(raw_folder: str = RAW_FOLDER) -> int | None:
    """
    Number of records the extraction that transform.py consumed says it saved.
    Mirrors transform.get_latest_extraction(): newest 'extraction_*' folder whose
    summary.json has status == 'completed'. Returns None if it can't be determined.
    """
    try:
        folders = sorted(
            (f for f in os.listdir(raw_folder)
             if f.startswith("extraction_") and os.path.isdir(os.path.join(raw_folder, f))),
            reverse=True,
        )
        for folder in folders:
            try:
                with open(os.path.join(raw_folder, folder, "summary.json"), encoding="utf-8") as fh:
                    summary = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if summary.get("status") == "completed":
                return int(summary["total_records"])
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def load_transformed_csv(path: str | Path = TRANSFORMED_CSV) -> pd.DataFrame:
    """
    Read transform.py's CSV with every column as text (no int/float guessing).

    Looks for `path` relative to the current folder first, then next to this
    script, then for a file with the same name next to this script. Uses
    utf-8-sig so a CSV that was opened and re-saved in Excel (which adds a BOM
    to the first header) still loads correctly.
    """
    path = Path(path)
    here = Path(__file__).resolve().parent
    candidates = [path, here / path, here / path.name]
    for candidate in candidates:
        if candidate.is_file():
            logger.info("Reading %s", candidate.resolve())
            return pd.read_csv(candidate, dtype=str, encoding="utf-8-sig")
    raise FileNotFoundError(
        f"Could not find '{path}'. Looked in:\n  "
        + "\n  ".join(str(c.resolve()) for c in candidates)
        + "\nRun transform.py first, or put launches_transformed.csv next to validation.py."
    )


if __name__ == "__main__":
    configure_logging()
    validate_launch_data(
        load_transformed_csv(),
        output_dir=PROCESSED_FOLDER,
        expected_records=get_expected_records(),
    )

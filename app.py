from pathlib import Path
import sqlite3
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page configuration and project paths
# --------------------------------------------------

st.set_page_config(
    page_title="Space Launch Analytics",
    page_icon="🚀",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "database" / "launches.db"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "validation_report.json"

REQUIRED_COLUMNS = [
    "launch_id",
    "launch_name",
    "launch_datetime",
    "launch_year",
    "launch_month",
    "provider",
    "rocket",
    "launch_location",
    "country",
    "status"
]

FINAL_SUCCESS = "Launch Successful"
FINAL_FAILURE = "Launch Failure"
FINAL_PARTIAL_FAILURE = "Launch was a Partial Failure"

FINAL_STATUSES = {
    FINAL_SUCCESS,
    FINAL_FAILURE,
    FINAL_PARTIAL_FAILURE
}

OUTCOME_COLORS = {
    "Success": "#2ca02c",
    "Partial failure": "#ff9f1c",
    "Failure": "#d62728",
    "Other known outcomes": "#9aa0a6"
}

OUTCOME_ORDER = list(OUTCOME_COLORS)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_launch_data(database_path, database_modified_time):
    """Read validated launch records from the SQLite database."""
    with sqlite3.connect(database_path) as connection:
        data = pd.read_sql_query("SELECT * FROM launches", connection)

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(
            "The launches table is missing these columns: "
            + ", ".join(sorted(missing_columns))
        )

    if data.empty:
        raise ValueError("The launches table contains no records.")

    data["launch_datetime"] = pd.to_datetime(
        data["launch_datetime"],
        errors="coerce",
        utc=True
    )

    data["launch_year"] = pd.to_numeric(
        data["launch_year"], errors="coerce"
    ).astype("Int64")

    data["launch_month"] = pd.to_numeric(
        data["launch_month"], errors="coerce"
    ).astype("Int64")

    text_columns = [
        "launch_name",
        "provider",
        "rocket",
        "launch_location",
        "country",
        "status"
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    return data


@st.cache_data
def load_validation_report(report_path, report_modified_time):
    """Read the validation report if it exists."""
    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def show_empty_message(message):
    st.info(message)


# --------------------------------------------------
# Analysis helpers (following the EDA notebook)
# --------------------------------------------------

def top_ten_with_other(data, column):
    counts = (
        data[column]
        .replace("", "Unknown")
        .fillna("Unknown")
        .value_counts()
    )

    top = counts.head(10)

    if len(counts) > 10:
        top.loc["Other"] = counts.iloc[10:].sum()

    return (
        top.sort_values(ascending=True)
        .rename_axis(column)
        .reset_index(name="launches")
    )


def outcome_group(status):
    value = str(status).strip().casefold()

    if value == FINAL_SUCCESS.casefold():
        return "Success"

    if value == FINAL_FAILURE.casefold():
        return "Failure"

    if value in {
        FINAL_PARTIAL_FAILURE.casefold(),
        "launch partial failure"
    }:
        return "Partial failure"

    return "Other known outcomes"


def make_outcome_chart(data, group_column, title, group_label):
    outcomes = data[data["status"].ne("Unknown")].copy()

    if outcomes.empty:
        show_empty_message("There are no known launch statuses for these filters.")
        return

    outcomes["outcome_group"] = outcomes["status"].apply(outcome_group)

    grouped = (
        outcomes.groupby([group_column, "outcome_group"])
        .size()
        .rename("launches")
        .reset_index()
    )

    grouped["group_total"] = grouped.groupby(group_column)["launches"].transform("sum")
    grouped["share"] = grouped["launches"] / grouped["group_total"] * 100

    fig = px.bar(
        grouped,
        x="share",
        y=group_column,
        color="outcome_group",
        orientation="h",
        barmode="stack",
        title=title,
        labels={
            "share": "Share of launches (%)",
            group_column: group_label,
            "outcome_group": "Recorded outcome"
        },
        custom_data=["launches", "group_total"],
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"outcome_group": OUTCOME_ORDER}
    )

    fig.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "Outcome: %{fullData.name}<br>"
            "Share: %{x:.1f}%<br>"
            "Count: %{customdata[0]} of %{customdata[1]}"
            "<extra></extra>"
        )
    )
    fig.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Load database
# --------------------------------------------------

st.title("🚀 Space Launch Analytics Dashboard")
st.write(
    "Explore validated historical rocket-launch records "
    "and the results of the ETL and EDA pipeline."
)

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.info(
        "Run your ETL pipeline, including validation.py and load.py, "
        "before starting the dashboard."
    )
    st.stop()

try:
    df = load_launch_data(
        str(DB_PATH),
        DB_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Could not load launch data: {error}")
    st.stop()

st.caption(f"Data source: `{DB_PATH.relative_to(PROJECT_ROOT)}`")
st.success(f"Loaded {len(df):,} validated launch records from SQLite.")


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

years = sorted(
    df["launch_year"].dropna().astype(int).unique().tolist()
)

providers = sorted(df["provider"].unique().tolist())

statuses = sorted(df["status"].unique().tolist())

selected_years = st.sidebar.multiselect(
    "Launch year",
    options=years,
    default=years
)

selected_providers = st.sidebar.multiselect(
    "Launch service provider",
    options=providers,
    default=providers
)

selected_statuses = st.sidebar.multiselect(
    "Launch status",
    options=statuses,
    default=statuses
)

filtered_df = df.copy()

if selected_years:
    filtered_df = filtered_df[
        filtered_df["launch_year"].isin(selected_years)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_providers:
    filtered_df = filtered_df[
        filtered_df["provider"].isin(selected_providers)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_statuses:
    filtered_df = filtered_df[
        filtered_df["status"].isin(selected_statuses)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]

# --------------------------------------------------
# Dashboard overview and KPI cards
# --------------------------------------------------

st.header("1. Launch Overview")

total_launches = len(filtered_df)
successful_launches = int(
    filtered_df["status"].eq(FINAL_SUCCESS).sum()
)
failed_launches = int(
    filtered_df["status"].eq(FINAL_FAILURE).sum()
)

partial_failures = int(
    filtered_df["status"].eq(FINAL_PARTIAL_FAILURE).sum()
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total launches", f"{total_launches:,}")
kpi2.metric("Successful launches", f"{successful_launches:,}")
kpi3.metric("Failed launches", f"{failed_launches:,}")
kpi4.metric("Partial failures", f"{partial_failures:,}")

if filtered_df.empty:
    st.warning("No records match the selected filters. Change the filters to see analysis.")
    st.stop()


# --------------------------------------------------
# Research Question 1: Launch activity over time
# --------------------------------------------------

st.header("2. Launch Activity Over Time")
st.subheader("How has the number of recorded launches changed by year and month?")

date_data = filtered_df.dropna(subset=["launch_datetime"]).copy()

if date_data.empty:
    show_empty_message("No valid launch dates are available for these filters.")
else:
    date_data["month_start"] = (
        date_data["launch_datetime"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        date_data.groupby("month_start")
        .size()
        .rename("launches")
    )

    all_months = pd.date_range(
        monthly.index.min(),
        monthly.index.max(),
        freq="MS"
    )

    monthly = (
        monthly.reindex(all_months, fill_value=0)
        .rename_axis("month_start")
        .reset_index(name="launches")
    )

    yearly = (
        filtered_df.dropna(subset=["launch_year"])
        .groupby("launch_year")
        .size()
        .reset_index(name="launches")
        .sort_values("launch_year")
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly,
            x="month_start",
            y="launches",
            markers=True,
            title="Monthly Recorded Launches",
            labels={
                "month_start": "Month",
                "launches": "Number of launches"
            }
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        yearly_display = yearly.copy()
        yearly_display["launch_year"] = yearly_display["launch_year"].astype(str)

        fig = px.bar(
            yearly_display,
            x="launch_year",
            y="launches",
            text="launches",
            title="Recorded Launches by Year",
            labels={
                "launch_year": "Year",
                "launches": "Number of launches"
            }
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Research Question 2: Most frequent providers, rockets, locations
# --------------------------------------------------

st.header("3. Providers, Rockets and Launch Locations")
st.subheader(
    "Which launch service providers, rocket configurations, "
    "and launch locations appear most frequently?"
)

category_columns = [
    ("provider", "Top 10 Launch Service Providers"),
    ("rocket", "Top 10 Rocket Configurations"),
    ("launch_location", "Top 10 Launch Locations")
]

chart_columns = st.columns(3)

for chart_column, (column, title) in zip(chart_columns, category_columns):
    with chart_column:
        chart_data = top_ten_with_other(filtered_df, column)

        fig = px.bar(
            chart_data,
            x="launches",
            y=column,
            orientation="h",
            text="launches",
            title=title,
            labels={
                column: column.replace("_", " ").title(),
                "launches": "Number of launches"
            }
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Research Question 3: Outcomes by provider and year
# --------------------------------------------------

st.header("4. Launch Outcomes")
st.subheader("How do recorded launch outcomes vary across providers and years?")

outcomes = filtered_df[filtered_df["status"].ne("Unknown")].copy()

if outcomes.empty:
    show_empty_message("No known launch outcomes are available for these filters.")
else:
    outcomes["outcome_group"] = outcomes["status"].apply(outcome_group)

    top_providers = outcomes["provider"].value_counts().head(10).index
    provider_outcomes = outcomes[
        outcomes["provider"].isin(top_providers)
    ].copy()

    provider_col, year_col = st.columns(2)

    with provider_col:
        make_outcome_chart(
            provider_outcomes,
            "provider",
            "Outcome Distribution by Top Providers (100%)",
            "Provider"
        )

    with year_col:
        year_outcomes = outcomes.dropna(subset=["launch_year"]).copy()
        year_outcomes["launch_year"] = year_outcomes["launch_year"].astype(str)

        make_outcome_chart(
            year_outcomes,
            "launch_year",
            "Outcome Distribution by Year (100%)",
            "Year"
        )


# --------------------------------------------------
# Research Question 4: Heatmap and year-over-year change
# --------------------------------------------------

st.header("5. Year-over-Year Changes and Monthly Patterns")
st.subheader(
    "What year-over-year changes and monthly patterns can be observed?"
)

year_month = (
    filtered_df.dropna(subset=["launch_year", "launch_month"])
    .groupby(["launch_year", "launch_month"])
    .size()
    .rename("launches")
    .reset_index()
)

if year_month.empty:
    show_empty_message("Year/month data is unavailable for these filters.")
else:
    heatmap_data = year_month.pivot(
        index="launch_year",
        columns="launch_month",
        values="launches"
    ).reindex(columns=range(1, 13)).fillna(0)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    heatmap_col, yoy_col = st.columns(2)

    with heatmap_col:
        fig = px.imshow(
            heatmap_data,
            labels={
                "x": "Month",
                "y": "Year",
                "color": "Number of launches"
            },
            x=month_names,
            y=heatmap_data.index.astype(str),
            text_auto=True,
            aspect="auto",
            title="Launch Activity Heatmap: Year × Month",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    annual_counts = (
        filtered_df.dropna(subset=["launch_year"])
        .groupby("launch_year")
        .size()
        .rename("launches")
        .reset_index()
        .sort_values("launch_year")
    )

    annual_counts["yoy_change_pct"] = (
        annual_counts["launches"].pct_change() * 100
    )

    yoy = annual_counts.dropna(subset=["yoy_change_pct"]).copy()

    with yoy_col:
        if yoy.empty:
            show_empty_message(
                "At least two years of launch records are needed to calculate year-over-year change."
            )
        else:
            yoy["direction"] = np.where(
                yoy["yoy_change_pct"] >= 0,
                "Increase",
                "Decrease"
            )
            yoy["launch_year"] = yoy["launch_year"].astype(str)

            fig = px.bar(
                yoy,
                x="launch_year",
                y="yoy_change_pct",
                color="direction",
                text=yoy["yoy_change_pct"].map(
                    lambda value: f"{value:+.1f}%"
                ),
                title="Year-over-Year Change in Launches",
                labels={
                    "launch_year": "Year",
                    "yoy_change_pct": "Change from previous year (%)",
                    "direction": "Change"
                }
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annual Launch Counts and Year-over-Year Change")
    st.dataframe(annual_counts, use_container_width=True, hide_index=True)


# --------------------------------------------------
# Research Question 5: Validation and data quality
# --------------------------------------------------

st.header("6. Data Quality Analysis")
st.subheader(
    "What proportion of source records contain missing, inconsistent, "
    "invalid, or duplicate data?"
)

if not VALIDATION_REPORT_PATH.exists():
    st.warning(
        "The validation report was not found. "
        "The database contains validated records, but source-level "
        "missing, invalid and duplicate counts cannot be reconstructed "
        "from this database alone."
    )
else:
    try:
        report = load_validation_report(
            str(VALIDATION_REPORT_PATH),
            VALIDATION_REPORT_PATH.stat().st_mtime
        )

        required_report_keys = [
            "total_records",
            "valid_records",
            "rejected_records",
            "duplicate_records",
            "missing_values",
            "invalid_records"
        ]

        missing_report_keys = [
            key for key in required_report_keys if key not in report
        ]

        if missing_report_keys:
            st.error(
                "The validation report is missing these expected fields: "
                + ", ".join(missing_report_keys)
            )
        else:
            total_extracted = int(report["total_records"])
            valid_count = int(report["valid_records"])
            rejected_count = int(report["rejected_records"])
            duplicates = int(report["duplicate_records"])
            missing_count = int(report["missing_values"])
            invalid_count = int(report["invalid_records"])

            after_dedup = total_extracted - duplicates

            q1, q2, q3, q4 = st.columns(4)

            q1.metric("Total extracted", f"{total_extracted:,}")
            q2.metric("Valid records", f"{valid_count:,}")
            q3.metric("Rejected records", f"{rejected_count:,}")
            q4.metric("Duplicate records", f"{duplicates:,}")

            if total_extracted > 0:
                p1, p2, p3 = st.columns(3)
                p1.metric(
                    "Valid records (%)",
                    f"{valid_count / total_extracted * 100:.2f}%"
                )
                p2.metric(
                    "Rejected records (%)",
                    f"{rejected_count / total_extracted * 100:.2f}%"
                )
                p3.metric(
                    "Duplicate records (%)",
                    f"{duplicates / total_extracted * 100:.2f}%"
                )

            quality_metrics = pd.DataFrame({
                "Metric": [
                    "Total extracted",
                    "After duplicate removal",
                    "Valid records",
                    "Rejected records",
                    "Duplicate records"
                ],
                "Count": [
                    total_extracted,
                    after_dedup,
                    valid_count,
                    rejected_count,
                    duplicates
                ]
            })

            issue_df = pd.DataFrame({
                "Category": [
                    "Valid",
                    "Missing",
                    "Invalid",
                    "Duplicate"
                ],
                "Records": [
                    valid_count,
                    missing_count,
                    invalid_count,
                    duplicates
                ]
            })

            quality_col, issue_col = st.columns(2)

            with quality_col:
                st.subheader("Validation Summary")
                st.dataframe(
                    quality_metrics,
                    use_container_width=True,
                    hide_index=True
                )

            with issue_col:
                st.subheader("Validation Results by Category")
                fig = px.bar(
                    issue_df,
                    x="Category",
                    y="Records",
                    text="Records",
                    labels={
                        "Category": "Validation category",
                        "Records": "Number of records"
                    }
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            if valid_count + rejected_count == total_extracted:
                st.success("Record accounting verified: valid + rejected = extracted.")
            else:
                st.warning(
                    "Record accounting does not match: valid + rejected "
                    "is different from total extracted."
                )

    except Exception as error:
        st.error(f"Could not read the validation report: {error}")


# Field completeness is calculated from the validated database.
st.subheader("Field Completeness in Validated Records")

completeness = (
    filtered_df.replace({"": np.nan, "Unknown": np.nan})
    .notna()
    .mean()
    .mul(100)
    .sort_values()
    .rename("Completeness (%)")
    .rename_axis("Field")
    .reset_index()
)

fig = px.bar(
    completeness,
    x="Completeness (%)",
    y="Field",
    orientation="h",
    text=completeness["Completeness (%)"].map(
        lambda value: f"{value:.1f}%"
    ),
    title="Field Completeness",
    labels={
        "Completeness (%)": "Completeness (%)",
        "Field": "Field"
    }
)
fig.update_xaxes(range=[0, 100])
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Field completeness describes validated database records. "
    "It does not replace source-level missingness because rejected records "
    "are not stored in the launches table."
)


# --------------------------------------------------
# Research Question 6: Provider success rates
# --------------------------------------------------

st.header("7. Provider Success Rates")
st.subheader(
    "Which providers have the highest recorded success rates "
    "among launches with known final outcomes?"
)

MIN_FINAL_LAUNCHES = 5

final_outcomes = filtered_df[
    filtered_df["status"].isin(FINAL_STATUSES)
].copy()

if final_outcomes.empty:
    show_empty_message(
        "No launches with known final outcomes are available for these filters."
    )
else:
    final_outcomes["is_success"] = (
        final_outcomes["status"] == FINAL_SUCCESS
    )

    provider_performance = (
        final_outcomes.groupby("provider")
        .agg(
            known_final_launches=("launch_id", "count"),
            successful_launches=("is_success", "sum")
        )
        .reset_index()
    )

    provider_performance["success_rate_pct"] = (
        provider_performance["successful_launches"]
        / provider_performance["known_final_launches"] * 100
    )

    ranked_providers = (
        provider_performance[
            provider_performance["known_final_launches"] >= MIN_FINAL_LAUNCHES
        ]
        .sort_values(
            ["success_rate_pct", "known_final_launches"],
            ascending=True
        )
    )

    if ranked_providers.empty:
        st.info(
            f"No provider has at least {MIN_FINAL_LAUNCHES} launches "
            "with known final outcomes. The minimum sample threshold "
            "matches the EDA notebook."
        )
    else:
        fig = px.bar(
            ranked_providers,
            x="success_rate_pct",
            y="provider",
            orientation="h",
            text=ranked_providers.apply(
                lambda row: (
                    f'{row["success_rate_pct"]:.1f}% '
                    f'(n={int(row["known_final_launches"])})'
                ),
                axis=1
            ),
            title=(
                "Provider Success Rates "
                f"(Minimum {MIN_FINAL_LAUNCHES} Final Outcomes)"
            ),
            labels={
                "success_rate_pct": "Success rate (%)",
                "provider": "Provider"
            },
            hover_data={
                "successful_launches": True,
                "known_final_launches": True,
                "success_rate_pct": ":.2f"
            }
        )
        fig.update_xaxes(range=[0, 100])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Provider Performance Details")
    st.dataframe(
        provider_performance.sort_values(
            ["success_rate_pct", "known_final_launches"],
            ascending=[False, False]
        ),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Launch records and CSV download
# --------------------------------------------------

st.header("8. Launch Records")
st.write("Browse and download the records matching the selected filters.")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered launch records (CSV)",
    data=csv_data,
    file_name="space_launch_data.csv",
    mime="text/csv"
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Space Launch Analytics | Data is read from the validated SQLite database. "
    "The dashboard does not call the launch API or modify the database."
)

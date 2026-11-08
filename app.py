
import streamlit as st
import pandas as pd
import json
import glob
import os

# Page configuration
st.set_page_config(
    page_title="Space Launch Analytics",
    page_icon="🚀",
    layout="wide"
)

# Dashboard title
st.title("🚀 Space Launch Analytics Dashboard")
st.write("Explore historical rocket launch data from 2020 to 2025.")

# Find the latest extraction folder
json_files = glob.glob("data/raw/extraction_*/page_*.json")

if not json_files:
    st.error("No launch data found!")
    st.stop()

# Load JSON data
launch_records = []

for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        launch_records.extend(data.get("results", []))

# Convert JSON into DataFrame
df = pd.json_normalize(launch_records)

st.success(f"Successfully loaded {len(df)} launch records.")
# KPI Cards
total_launches = len(df)

successful_launches = (
    df["status.name"] == "Launch Successful"
).sum()

failed_launches = (
    df["status.name"] == "Launch Failure"
).sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Launches", total_launches)
col2.metric("Successful Launches", successful_launches)
col3.metric("Failed Launches", failed_launches)
# Launch Status Chart
st.subheader("🚀 Launch Status Distribution")

status_counts = df["status.name"].value_counts()

st.bar_chart(status_counts)
# Launch Status Filter
st.sidebar.header("🔎 Dashboard Filters")

statuses = ["All"] + sorted(df["status.name"].dropna().unique().tolist())

selected_status = st.sidebar.selectbox(
    "Select Launch Status",
    statuses
)

if selected_status == "All":
    filtered_df = df
else:
    filtered_df = df[df["status.name"] == selected_status]

# Launch Provider Filter
providers = ["All"] + sorted(
    df["launch_service_provider.name"].dropna().unique().tolist()
)

selected_provider = st.sidebar.selectbox(
    "Select Launch Provider",
    providers
)

if selected_provider != "All":
    filtered_df = filtered_df[
        filtered_df["launch_service_provider.name"] == selected_provider
    ]
# Launch Year Filter
df["launch_year"] = pd.to_datetime(
    df["net"], errors="coerce"
).dt.year

years = sorted(
    df["launch_year"].dropna().astype(int).unique().tolist()
)

selected_year = st.sidebar.selectbox(
    "Select Launch Year",
    ["All"] + years
)

# Apply year filter
if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["net"].notna() &
        (pd.to_datetime(filtered_df["net"], errors="coerce").dt.year == selected_year)
    ]
# Rocket Configuration Chart
st.subheader("🚀 Rocket Configuration Distribution")

rocket_counts = (
    filtered_df["rocket.configuration.name"]
    .value_counts()
)

st.bar_chart(rocket_counts)
# Launches by Year
st.subheader("📅 Launches by Year")

year_counts = (
    filtered_df["net"]
    .dropna()
    .pipe(pd.to_datetime, errors="coerce")
    .dropna()
    .dt.year
    .value_counts()
    .sort_index()
)

st.bar_chart(year_counts)
# Launches by Provider
st.subheader("🏢 Launches by Provider")

provider_counts = (
    filtered_df["launch_service_provider.name"]
    .value_counts()
)

st.bar_chart(provider_counts)
# Download Filtered Data
st.subheader("📥 Download Launch Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="space_launch_data.csv",
    mime="text/csv"
)
# Filtered Data Summary
st.subheader("📊 Filtered Data Summary")

total_filtered = len(filtered_df)

successful = (
    filtered_df["status.name"] == "Launch Successful"
).sum()

success_rate = (
    (successful / total_filtered) * 100
    if total_filtered > 0 else 0
)

col1, col2 = st.columns(2)

col1.metric("Filtered Launches", total_filtered)
col2.metric("Success Rate", f"{success_rate:.1f}%")
# Data Quality Analysis
st.subheader("🔍 Missing Values Analysis")

missing_data = filtered_df.isnull().sum().reset_index()

missing_data.columns = [
    "Column Name",
    "Missing Values"
]

st.dataframe(
    missing_data,
    use_container_width=True
)
# Launch Timeline
st.subheader("📅 Launch Timeline")

timeline_df = filtered_df.copy()

timeline_df["net"] = pd.to_datetime(
    timeline_df["net"], errors="coerce"
)

timeline_df = timeline_df.dropna(subset=["net"])

if not timeline_df.empty:

    timeline_counts = (
        timeline_df
        .groupby(timeline_df["net"].dt.strftime("%Y-%m-%d"))
        .size()
        .sort_index()
    )

    st.line_chart(timeline_counts)

else:
    st.warning("No valid launch dates available for the selected filters.")
# Preview data
st.subheader("Launch Records")
st.dataframe(filtered_df, use_container_width=True)

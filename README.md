# SpaceLaunch_Analytics_Project
An end-to-end ETL and Exploratory Data Analysis (EDA) project analyzing historical orbital and recorded rocket launches from 2020–2025 using Python, APIs, Pandas, Pydantic, and SQLite.

## Project Structure

```text
SpaceLaunch_Analytics/
│
├── data/
│   ├── raw/             # Original API data (JSON)
│   ├── processed/       # Cleaned and transformed data
│   └── launches.db      # SQLite database
│
├── src/                 # ETL pipeline scripts
│   ├── extract.py       # Fetches data from the API
│   ├── transform.py     # Cleans and standardizes data
│   ├── validate.py      # Validates data using Pydantic
│   ├── load.py          # Loads data into storage
│   └── database.py      # Handles SQLite operations
│
├── notebooks/
│   └── eda.ipynb        # Exploratory Data Analysis
│
├── logs/
│   └── validation.log   # Pipeline and validation logs
│
├── tests/
│   └── test_pipeline.py # Automated pipeline tests
│
├── app.py               # Streamlit dashboard
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── .gitignore           # Files excluded from Git
```

### Folder Descriptions

- **data/** – Stores raw API responses, processed datasets, and the SQLite database.
- **src/** – Contains the Python scripts responsible for extraction, transformation, validation, and loading (ETL).
- **notebooks/** – Contains Jupyter notebooks for exploratory data analysis and visualization.
- **logs/** – Stores pipeline execution and data validation logs.
- **tests/** – Contains automated tests to verify pipeline functionality.

### Main Files

- **app.py** – Runs the Streamlit analytics dashboard.
- **requirements.txt** – Lists all required Python libraries.
- **README.md** – Provides project documentation and setup instructions.
- **.gitignore** – Specifies files and folders Git should ignore.

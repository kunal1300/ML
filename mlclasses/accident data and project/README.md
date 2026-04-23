# Indian Roads Accident Analysis And Dashboard

This repository contains an exploratory data analysis notebook and a Streamlit dashboard for the Indian roads accident dataset. The project studies accident patterns across city, state, date, time, road type, weather, visibility, traffic density, cause, severity, casualties, vehicles involved, and risk score.

## Project Files

- `app.py` - Streamlit dashboard for interactive accident analysis.
- `notebooks/Indian_Roads_EDA.ipynb` - Jupyter notebook with exploratory data analysis.
- `data/indian_roads_dataset.csv` - Dataset used by the dashboard.
- `requirements.txt` - Python packages needed to run the dashboard.

## Important Work Completed

- Imported and explored the Indian roads accident dataset.
- Checked dataset shape, columns, data types, missing values, and summary statistics.
- Cleaned and prepared the data for analysis.
- Converted date fields into useful time-based features such as year, month, month name, hour, weekend, and peak-hour indicators.
- Analyzed numerical features including temperature, lanes, vehicles involved, casualties, and risk score.
- Analyzed categorical features such as city, state, road type, weather, visibility, traffic density, cause, and accident severity.
- Studied hidden accident patterns by peak hours, weekday/weekend behavior, and time of day.
- Compared weather and environmental factors such as fog, rain, visibility, and temperature.
- Reviewed road and infrastructure factors such as road type, number of lanes, and traffic signals.
- Examined accident causes and severity levels, including minor, major, and fatal accidents.
- Identified geographic hotspots by city and state.
- Explored risk factors and correlations to understand high-risk accident conditions.
- Built an interactive dashboard with filters, KPIs, charts, heatmap, map, filtered data table, and CSV download.
- Improved the dashboard design with a hero header, styled KPI cards, readable charts, data labels, and better color contrast.

## Libraries And Imports Used

The EDA notebook uses:

- `pandas` for data loading, cleaning, grouping, and analysis.
- `numpy` for numerical operations.
- `matplotlib.pyplot` for static visualizations.
- `seaborn` for statistical charts.
- `plotly.express`, `plotly.graph_objects`, and `plotly.subplots.make_subplots` for interactive visualizations.
- `scipy.stats` and `chi2_contingency` for statistical analysis.
- `warnings` to manage notebook warnings.

The Streamlit dashboard uses:

- `streamlit` for the web dashboard.
- `pandas` for data processing.
- `altair` for interactive dashboard charts.
- `openpyxl` for Excel file upload support.

## Dashboard Features

- Sidebar filters for date, state, city, severity, road type, weather, cause, traffic density, week type, and peak-hour status.
- KPI cards for accidents, casualties, fatal accidents, average risk, and peak-hour share.
- Monthly trend chart for accidents, casualties, and average risk.
- Severity mix chart.
- City hotspot ranking.
- Cause-by-severity stacked chart.
- Day-and-hour risk heatmap.
- Road and weather profile chart.
- Accident location map using latitude and longitude.
- Filtered records table and download button.

## Run The Dashboard

Install the requirements:

```powershell
pip install -r requirements.txt
```

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The app loads `data/indian_roads_dataset.csv` by default. You can also upload a CSV or Excel file from the sidebar.

## Recommended Dataset Columns

- `accident_id`
- `date`
- `time`
- `city`
- `state`
- `latitude`
- `longitude`
- `hour`
- `day_of_week`
- `is_weekend`
- `road_type`
- `lanes`
- `traffic_signal`
- `weather`
- `visibility`
- `temperature`
- `traffic_density`
- `cause`
- `accident_severity`
- `vehicles_involved`
- `casualties`
- `is_peak_hour`
- `festival`
- `risk_score`

# Forecast-app
Streamlit application for forecasting data from a file. It will allow users to upload a CSV, select columns for forecasting or use an "auto" mode, train a fast model, and display charts with a dotted forecast line extending from historical data.

## Data Input
### CSV Upload
Use the sidebar to upload a CSV file (e.g., with columns like date, sales).

### Column Selection
Manual: Pick the date and value columns from dropdowns.
Auto: The app detects a date column (most values parse as dates) and a numeric value column.

## Forecasting
### Data Prep
The selected columns are renamed to ds (date) and y (value), with dates converted to datetime format.

### Model
Prophet trains quickly on the data, handling trends and seasonality.

### Forecast
Predicts up to 365 periods ahead, with the user adjusting the visible forecast length via a slider.

## Visualization
### Chart Features
Historical Data: Solid blue line.
Forecast: Dotted orange line starting where historical data ends.
Confidence Intervals: Light orange shaded area around the forecast.
Forecast Start: Dashed green vertical line.
Fitted Values (optional): Purple line showing the model’s fit on historical data.
Interactivity: Zoom, pan, and hover over points with Plotly’s built-in tools.
Styling: Clean white background, clear labels, and modern colors inspired by Bravo Research’s professional look.

### Additional Features
Slider: Adjust forecast periods (1 to 365) dynamically.
Table: View forecast values with confidence intervals.
Download: Export the forecast as a CSV.


## Example Usage

### Upload a CSV
A file with date (e.g., "2023-01-01") and sales for example (e.g., 100.5).

### Select Columns
Choose date as the date column and sales as the value, or use "Auto."

### Adjust Forecast
Slide to 60 days to see a two-month prediction.

### View Results
See the chart with historical data, a dotted forecast, and download the result.


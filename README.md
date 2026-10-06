# Telecom Customer Churn Analysis - YuvaIntern Week 1

## Objective
Explore a publicly available telecom customer dataset, identify data-quality issues, perform cleaning/preprocessing, engineer simple features, and communicate useful telecom business insights with Python.

## Dataset
IBM Telco Customer Churn sample dataset (7,043 customer records, 21 raw columns).

## Files
- `telecom_churn_analysis.py` — reproducible Python workflow
- `README.md` — project documentation
- `telco_customer_churn_cleaned.csv` — generated after running the script

## Libraries
```bash
pip install pandas numpy matplotlib seaborn
```

## Run
```bash
python telecom_churn_analysis.py
```

The script downloads the public IBM sample dataset, performs data-quality checks, converts `TotalCharges` to numeric, handles missing values, encodes churn, creates tenure/service features, prints EDA summaries, creates charts, and saves a cleaned CSV.

## Business problem
Customer churn reduces recurring telecom revenue and increases customer-acquisition costs. The analysis investigates which customer characteristics are associated with higher churn so a telecom company can prioritize retention actions.

## Important limitation
The dataset is a cross-sectional sample. Observed relationships are associations, not proof that a specific factor causes churn.

# Hybrid Predictive Maintenance using Enhanced CMAPSS NASA Dataset

## Overview & Motivation

Traditional predictive maintenance systems often focus on predicting a binary outcome: whether a machine will fail or not. These systems, while useful, lack granularity and may fail to provide early warnings when a component begins to degrade. In critical systems like aircraft engines or turbines, degradation typically happens gradually and goes through multiple phases before catastrophic failure occurs. Binary models overlook these intermediate warning stages, which could be used to plan better maintenance schedules and avoid downtime.

This project aims to bridge the gap between theoretical predictive models and practical industrial needs. We move beyond binary failure prediction by:

- Introducing multi-stage fault progression to reflect gradual degradation (5 levels: Normal → Slightly Degraded → Moderately Degraded → Critical → Failure).
- Generating custom labels from raw sensor data using clustering, rather than relying on CMAPSS’s standard labels.
- Combining classification and regression approaches to simultaneously predict the current health stage and estimate how long the system will stay in it.
- Calculating a Risk Score to assist in maintenance decision-making.

These steps reflect how predictive maintenance is handled in modern industrial applications and will help gain deeper insights into system behavior over time.

## Dataset

This project uses the **Enhanced CMAPSS NASA Dataset**.

- **Link to the dataset:** [https://www.kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- The dataset consists of turbofan engine degradation simulation run-to-failure data. It includes multiple multivariate time series, with each data set further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type.

## Setup and Installation

### Prerequisites

- Python 3.10.0 (as specified in `.python-version`)
- Bash (for running `.sh` scripts)

### Dependencies

The project dependencies are listed in the `pyproject.toml` file and include:

```toml
dependencies = [
    "pandas",
    "scikit-learn",
    "numpy",
    "matplotlib",
    "seaborn",
    "imbalanced-learn", # For SMOTE
    "xgboost",
    "lightgbm",
    # ... other dependencies
]
```

### Installation

1. Clone the repository:

```
git clone <repository_url>
cd <repository_name>
```

2. Create a virtual environment:

```
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:

   ```
   venv\Scripts\activate
   ```

   - On macOS/Linux:

   ```
   source venv/bin/activate
   ```

4. Install the dependencies:

```
pip install -r requirements.txt
```

5. Install the project in editable mode:

```
pip install -e .
```

6. Run each script from scripts folder:

```
bash scripts/download_data.sh
bash scripts/setup_database.sh
python scripts/run_eda_feature_eng.py
python scripts/clustering.py
python scripts/classification.py
python scripts/regression.py
python scripts/risk_assessment.py
```

# Contibution

The contributions for the project are:

    - C. V. Rishi ( se23ucse044 ) - 40%
    - Pranav Verma (se23ucse154) - 15%
    - Somu ( se23ucse140 ) - 15%
    - Saatwik Maddhireddy ( se23ucse155 ) - 15%
    - Monish Pratapani ( se23ucse112 ) - 15%

# PG&E Energy Analytics Challenge
This repository contains the code, data, and analysis developed for the 2025 PG&E Energy Analytics Challenge. Our goal is to forecast hourly electricity load for a full calendar year in a designated California region using historical data and exogenous environmental variables. The project leverages machine learning techniques—most notably, the XGBoost algorithm—to capture complex non-linear interactions between electricity load, temperature, and Global Horizontal Irradiance (GHI).

## Overview
The challenge focuses on accurately predicting electricity demand under realistic conditions, where forecasts for any given day must be generated using only data available up to that time. Our methodology includes:

* **Data Exploration & Preprocessing**:
Analyzing hourly load data alongside temperature and GHI from multiple sites. We employ harmonic transformations to capture cyclic patterns and aggregate weather data to account for regional effects.

* **Model Selection & Training**:
Implementing an XGBoost regressor that is fine-tuned via time-series cross-validation and grid search over key hyperparameters. This model is chosen for its robustness in modeling complex, non-linear relationships.

* **Model Evaluation & Forecast Generation**:
Evaluating performance using metrics such as MAE, MSE, and MAPE. The final model produces reliable forecasts that mimic observed diurnal and seasonal trends.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── datasets
│   ├── results.xlsx
│   ├── testing.xlsx
│   └── training.xlsx
├── final_report.pdf
├── main.py
├── notebooks
│   ├── eda.ipynb
│   └── interactive_eda.ipynb
├── requirements.txt
└── utils
    ├── visualization.py
    └── xgboost.py
```

## Installation
1. Clone the Repository:

```
git clone https://github.com/vohuynhquangnguyen/2025-PGE-Energy-Analytics.git
cd 2025-PGE-Energy-Analytics
```

2. Set Up a Virtual Environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install Dependencies:

```
pip install -r requirements.txt
```

## Usage
### Interactive Analysis
Open the Jupyter notebooks in the notebooks directory to explore the data, visualize key insights, and review the model development process:

```
jupyter ./notebooks/interactive_eda.ipynb
```

### Running the Scripts
Use the following command to run:

```
python main.py
```

### Methodology
Our approach can be summarized as follows:
* **Data Preprocessing**:
We standardize features and create new ones to capture temporal patterns (e.g., sine/cosine transformations of the hour) and aggregate multi-site weather data.

* **Modeling**:
The XGBoost algorithm is employed due to its ability to model non-linear relationships and its built-in regularization. Model hyperparameters (number of trees, tree depth, learning rate, etc.) are fine-tuned using a time-series aware cross-validation strategy.

* **Validation**:
Model performance is evaluated using metrics such as MAE, MSE, and MAPE, ensuring the forecasts are both accurate and robust.

## License
The project is licensed under the MIT License.
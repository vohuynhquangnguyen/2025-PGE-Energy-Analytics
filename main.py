from utils.visualization import run_exploratory_analysis
from utils.xgboost import run_forecasting_pipeline

def main():
    print("Running Exploratory Analysis...")
    run_exploratory_analysis()
    
    print("\nRunning Model Training, Validation, and Forecasting...")
    run_forecasting_pipeline()

if __name__ == "__main__":
    main()

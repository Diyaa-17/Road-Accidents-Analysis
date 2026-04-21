from data_downloader import download_data
from data_preprocessing import clean_data, preprocess_features
from eda import perform_eda
from statistical_analysis import perform_statistical_analysis
# from model_training import build_model, build_linear_model
import warnings
warnings.filterwarnings("ignore")

def main():
    print("="*60)
    print(" ROAD ACCIDENT PIPELINE: CRISP-DM METHODOLOGY")
    print("="*60)
    
    print("\n--- CRISP-DM Phase 1: Business Understanding ---")
    print("Objective: Predict road accident severity and interpret conditional factors using historical data.")

    print("\n--- CRISP-DM Phase 2: Data Understanding ---")
    # Data Acquisition via SQLite & CSV
    data_path = download_data()
    
    # Needs cleaning before EDA
    print("\n--- CRISP-DM Phase 3: Data Preparation ---")
    df_raw = clean_data(data_path)
    df_processed = preprocess_features(df_raw)

    print("\n--- Returning to CRISP-DM Phase 2: Data Understanding (Analysis) ---")
    # Statistical Analysis
    perform_statistical_analysis(df_processed)
    # Exploratory Data Analysis
    perform_eda(df_processed)
    
    # Model training step skipped as requested

    print("="*60)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY ")
    print("="*60)

if __name__ == "__main__":
    main()

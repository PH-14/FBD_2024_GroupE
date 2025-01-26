import argparse
import pandas as pd
from data_preprocessing import preprocessing
from clustering import create_clusters

def main(run_full_analysis):
    if run_full_analysis:
        print("Running full analysis...")
        data_path = "data/period_data.tar"
    else:
        print("Running demo workflow...")
        data_path = "data/demo_data.tar"

    # Step 1: Load and Preprocess Data
    print("Preprocessing data...")
    processed_data = preprocessing(data_path, demo = run_full_analysis)

    # Step 2: Create our Clusters
    print("Running analysis...")
    cluster_results = create_clusters(processed_data)

    # Step 3: Run our strategies
    print("Saving results...")
    save_results(analysis_results, output_path="results/analysis_results.csv")

    print("Workflow completed successfully.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run full analysis or demo workflow.")
    parser.add_argument(
        "run_full_analysis", 
        type=bool, 
        nargs="?", 
        const=True, 
        default=False,
        help="Set to True to run the full analysis, or False to run the demo workflow (default: False)"
    )
    args = parser.parse_args()

    # Call the main function with the parsed argument
    main(run_full_analysis=args.run_full_analysis)
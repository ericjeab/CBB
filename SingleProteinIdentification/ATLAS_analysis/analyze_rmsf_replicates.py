"""
Calculate average correlation and variance (spread) of RMSF values from the ATLAS dataset.

This script identifies the 150 smallest proteins from an information CSV, downloads 
their RMSF data from the ATLAS API, and calculates the global average standard 
deviation and average Pearson correlation between the independent simulation runs.

Expected Input: '2024_11_18_ATLAS_info.csv'
Author: Eric Jeanbourquin
Date: 2026-05-01
"""

import io
import zipfile
from itertools import combinations
from tqdm import tqdm

import numpy as np
import pandas as pd
import requests

# --- Configuration Constants ---
# Putting these at the top makes it easy to change them later without hunting through code
API_URL = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/"
INFO_FILE = "2024_11_18_ATLAS_info.csv"
NUM_PROTEINS = 150
RMSF_COLUMNS = ['RMSF_R1', 'RMSF_R2', 'RMSF_R3']

def main():
    # 1. Read in metadata and identify target proteins
    print(f"Loading metadata from {INFO_FILE}...")
    df_info = pd.read_csv(INFO_FILE, sep="\t")
    
    df_sorted = df_info.sort_values('length', ascending=False)
    top_proteins = df_sorted['PDB'].astype(str).head(NUM_PROTEINS).tolist()

    spread_list = []
    correlation_list = []

    # 2. Process each protein
    print(f"Processing {NUM_PROTEINS} proteins. This may take a moment...")
    
    pbar = tqdm(top_proteins,desc="Downloading",ncols=100)

    for protein_name in pbar:
        pbar.set_postfix({'Current Protein': protein_name})

        response = requests.get(API_URL + protein_name)

        if response.status_code == 200:
            # Load zip file into memory (saves disk space and runs faster!)
            zip_data = zipfile.ZipFile(io.BytesIO(response.content))
            target_file = f"{protein_name}_RMSF.tsv"

            if target_file in zip_data.namelist():
                # Read the TSV directly from the zip file in memory
                with zip_data.open(target_file) as file:
                    df_rmsf = pd.read_csv(file, sep='\t')
                
                # Calculate average standard deviation (spread) across runs
                average_spread = np.std(df_rmsf[RMSF_COLUMNS], axis=1).mean()
                
                # Calculate average pairwise correlation
                correlations = [df_rmsf[a].corr(df_rmsf[b]) for a, b in combinations(RMSF_COLUMNS, 2)]
                average_correlation = np.mean(correlations)
                
                # Store results
                spread_list.append(average_spread)
                correlation_list.append(average_correlation)
            else:
                # If we MUST print a warning, tqdm.write() safely prints above the bar without breaking it
                tqdm.write(f"  Warning: '{target_file}' not found inside zip for {protein_name}.")
        else:
            tqdm.write(f"  Download Failed for {protein_name}! Status Code: {response.status_code}")

    # 3. Calculate and display the final global metrics
    if spread_list and correlation_list:
        global_avg_spread = np.mean(spread_list)
        global_avg_corr = np.mean(correlation_list)
        
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print(f"Successfully processed: {len(spread_list)} proteins.")
        print(f"Global Avg RMSF Spread (Std Dev): {global_avg_spread:.4f} Å")
        print(f"Global Avg RMSF Correlation:      {global_avg_corr:.4f}")
    else:
        print("\nNo data was successfully processed.")

# This ensures the script runs when called directly from the terminal
if __name__ == "__main__":
    main()

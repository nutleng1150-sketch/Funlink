#!/usr/bin/env python3
import argparse
import sqlite3
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


# ---------------------------------------------------------
# Fix NaN / Inf in probability matrices
# ---------------------------------------------------------
def fix_nan_scores(scores, label_name=""):
    scores = np.array(scores, dtype=float)
    n_samples, n_classes = scores.shape

    bad_rows = ~np.isfinite(scores).all(axis=1)
    if bad_rows.sum() > 0:
        print(f" {bad_rows.sum()} NaN/inf rows in {label_name}, fixing...")
        scores[bad_rows] = 1.0 / n_classes

    row_sums = scores.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).squeeze()
    if zero_rows.sum() > 0:
        print(f" {zero_rows.sum()} zero-probability rows in {label_name}, fixing...")
        scores[zero_rows] = 1.0 / n_classes
        row_sums = scores.sum(axis=1, keepdims=True)

    scores = scores / row_sums
    return scores


# ---------------------------------------------------------
# Load label maps from metadata
# ---------------------------------------------------------
def load_label_maps(metadata_file):
    df = pd.read_csv(metadata_file, sep="\t")

    species_names = sorted(df["species"].unique())
    condition_names = sorted(df["condition"].unique())

    species_map = {i: name for i, name in enumerate(species_names)}
    condition_map = {i: name for i, name in enumerate(condition_names)}

    return species_map, condition_map, species_names, condition_names


# ---------------------------------------------------------
# Predict a single genome (one table)
# ---------------------------------------------------------
def predict_single_table(table_name, db_file, model_file):
    print(f" Loading data from table: {table_name}")

    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    if df.empty:
        raise ValueError(f" Table '{table_name}' is empty or does not exist!")

    print(" Loading trained model...")
    model_species, model_condition, scaler, encoder = joblib.load(model_file)

    print(" Encoding & scaling features...")
    X_encoded = encoder.transform(df)
    X_scaled = scaler.transform(X_encoded.toarray())

    proba_species = model_species.predict_proba(X_scaled)
    proba_condition = model_condition.predict_proba(X_scaled)

    proba_species = fix_nan_scores(proba_species, "Species")
    proba_condition = fix_nan_scores(proba_condition, "Condition")

    # Mean probabilities (genome-level)
    mean_species = proba_species.mean(axis=0)
    mean_condition = proba_condition.mean(axis=0)

    species_idx = int(np.argmax(mean_species))
    condition_idx = int(np.argmax(mean_condition))

    return (
        species_idx,
        mean_species,
        condition_idx,
        mean_condition,
        df.shape[0],
    )


# ---------------------------------------------------------
# Generate HTML Report
# ---------------------------------------------------------
def save_html_report(
    table_name,
    species_idx,
    species_map,
    species_probs,
    condition_idx,
    condition_map,
    condition_probs,
    n_rows,
    output_html,
):
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Prediction Report – {table_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 60%; margin-top: 10px; }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f4f4f4;
            font-weight: bold;
        }}
        .highlight {{
            background-color: #d5f5e3;
            font-weight: bold;
        }}
    </style>
</head>
<body>

<h1>Funlink ML Prediction Report</h1>
<p><b>Sample:</b> {table_name}</p>
<p><b>Computed on:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><b>Total Feature Rows:</b> {n_rows}</p>

<h2>Predicted Species</h2>
<p><b>Final Call:</b> {species_map[species_idx]}</p>
<table>
<tr><th>Species</th><th>Probability</th></tr>
"""

    for i, p in enumerate(species_probs):
        highlight = "class='highlight'" if i == species_idx else ""
        html += f"<tr {highlight}><td>{species_map[i]}</td><td>{p:.4f}</td></tr>"

    html += """
</table>

<h2>Predicted Condition</h2>
<p><b>Final Call:</b> """ + condition_map[condition_idx] + """</p>
<table>
<tr><th>Condition</th><th>Probability</th></tr>
"""

    for i, p in enumerate(condition_probs):
        highlight = "class='highlight'" if i == condition_idx else ""
        html += f"<tr {highlight}><td>{condition_map[i]}</td><td>{p:.4f}</td></tr>"

    html += """
</table>

<br><br>
<hr>
<p><small>Generated automatically by Funlink ML Prediction Engine.</small></p>
</body>
</html>
"""

    with open(output_html, "w") as f:
        f.write(html)

    print(f" HTML report saved to: {output_html}")


# ---------------------------------------------------------
# Main CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict species & condition for a genome using Funlink ML model"
    )
    parser.add_argument("--db_file", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--table_name", required=True)
    parser.add_argument("--html_out", default="prediction_output.html",
                        help="Output HTML file name")
    args = parser.parse_args()

    # Load label mappings
    species_map, condition_map, species_list, condition_list = load_label_maps(
        args.metadata_file
    )

    species_idx, species_probs, condition_idx, condition_probs, n_rows = predict_single_table(
        args.table_name, args.db_file, args.model_file
    )

    # Normal terminal output
    print("\n============================")
    print("     FINAL PREDICTION")
    print("============================")
    print(f" Genome contained {n_rows} rows/features\n")

    print(" Species Prediction:")
    print(f"  → Predicted Species: {species_map[species_idx]}")
    for i, p in enumerate(species_probs):
        print(f"     {species_map[i]:30s} {p:.4f}")

    print("\n Condition Prediction:")
    print(f"  Predicted Condition: {condition_map[condition_idx]}")
    for i, p in enumerate(condition_probs):
        print(f"     {condition_map[i]:30s} {p:.4f}")

    # Save HTML output
    save_html_report(
        args.table_name,
        species_idx,
        species_map,
        species_probs,
        condition_idx,
        condition_map,
        condition_probs,
        n_rows,
        args.html_out,
    )

    print("\n Prediction completed & HTML report generated!\n")


if __name__ == "__main__":
    main()

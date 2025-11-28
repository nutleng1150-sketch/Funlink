#!/usr/bin/env python3
import argparse
import sqlite3
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support
)

from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------------------------------------
# Load tables safely
# -------------------------------------------------------------
def get_table_names(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [t[0] for t in cur.fetchall()]

# -------------------------------------------------------------
# Confusion Matrix Plotter
# -------------------------------------------------------------
def plot_confusion_matrix(cm, class_list, title, filename, normalize=False):
    plt.figure(figsize=(8, 6))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_list,
        yticklabels=class_list
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" Confusion matrix saved: {filename}")

# -------------------------------------------------------------
# Precision–Recall Curve (multi-class, safe)
# -------------------------------------------------------------
def plot_multiclass_pr(y_true_num, y_scores, class_list, title, filename):
    n_classes = len(class_list)
    n_score_cols = y_scores.shape[1]

    # Guard: shape mismatch or too few classes
    if n_classes < 2:
        print(f" Skipping PR plot '{title}' because there is only {n_classes} class.")
        return

    if n_score_cols != n_classes:
        print(
            f" Skipping PR plot '{title}' because y_scores has "
            f"{n_score_cols} columns but there are {n_classes} classes."
        )
        return

    y_bin = label_binarize(y_true_num, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f"{class_list[i]} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" PR Curve saved: {filename}")

# -------------------------------------------------------------
# Class Distribution Plot
# -------------------------------------------------------------
def plot_class_distribution(true_labels, class_list, title, filename):
    counts = pd.Series(true_labels).value_counts().reindex(class_list)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" Class distribution saved: {filename}")

# -------------------------------------------------------------
# Fix NaN / Inf scores
# -------------------------------------------------------------
def fix_nan_scores(scores, label_name=""):
    scores = np.array(scores, dtype=float)
    n_samples, n_classes = scores.shape

    # Bad rows (NaN/inf)
    bad_rows = ~np.isfinite(scores).all(axis=1)
    if bad_rows.sum() > 0:
        print(f" {bad_rows.sum()} NaN/inf rows found in {label_name}. Fixing.")
        scores[bad_rows] = 1.0 / n_classes

    # Zero-sum rows
    row_sums = scores.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).squeeze()
    if zero_rows.sum() > 0:
        print(f" {zero_rows.sum()} zero-sum rows found in {label_name}. Fixing.")
        scores[zero_rows] = 1.0 / n_classes
        row_sums = scores.sum(axis=1, keepdims=True)

    scores = scores / row_sums
    return scores

# -------------------------------------------------------------
# Metric Summary Heatmap
# -------------------------------------------------------------
def plot_metric_summary_heatmap(metrics_dict, title, filename):
    df = pd.DataFrame(metrics_dict)

    plt.figure(figsize=(7, 4))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f",
                cbar=True, linewidths=.5)

    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" Metric summary heatmap saved: {filename}")

# -------------------------------------------------------------
# Bar plot for metrics per task
# -------------------------------------------------------------
def plot_metric_bar(metrics, title, filename):
    plt.figure(figsize=(6, 4))
    names = list(metrics.keys())
    values = list(metrics.values())

    sns.barplot(x=names, y=values, palette="Set2")
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" Bar plot saved: {filename}")

# -------------------------------------------------------------
# Per-class metric heatmap
# -------------------------------------------------------------
def plot_class_metric_heatmap(y_true, y_pred, class_list, title, filename):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_list)), zero_division=0
    )

    df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }, index=class_list)

    plt.figure(figsize=(7, 5))
    sns.heatmap(df, annot=True, cmap="Oranges", fmt=".3f", linewidths=.5)
    plt.title(title)
    plt.xlabel("Metric")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f" Class metric heatmap saved: {filename}")

# -------------------------------------------------------------
# PDF Report Generator
# -------------------------------------------------------------
def create_pdf_report(output_pdf):
    figures = [
        "cm_species.png", "cm_species_norm.png",
        "pr_species.png", "dist_species.png",
        "cm_condition.png", "cm_condition_norm.png",
        "pr_condition.png", "dist_condition.png",
        "metric_summary_heatmap.png",
        "metric_bar_species.png", "metric_bar_condition.png",
        "species_class_metrics.png", "condition_class_metrics.png"
    ]

    with PdfPages(output_pdf) as pdf:
        for fig in figures:
            try:
                img = plt.imread(fig)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig()
                plt.close()
                print(f" Added to PDF: {fig}")
            except FileNotFoundError:
                print(f" Missing figure skipped: {fig}")

    print(f"\n PDF report created: {output_pdf}")

# -------------------------------------------------------------
# Main Evaluation
# -------------------------------------------------------------
def evaluate(db_file, metadata_file, model_file, batch_size=5000):

    print(" Loading trained model...")
    model_species, model_condition, scaler, encoder = joblib.load(model_file)

    conn = sqlite3.connect(db_file)
    metadata = pd.read_csv(metadata_file, sep="\t")
    tables = get_table_names(conn)

    true_species = []
    true_condition = []
    score_species_list = []
    score_condition_list = []

    print(f" Found {len(tables)} tables.")

    for table_name in tables:
        print(f" Evaluating: {table_name}")

        meta = metadata[metadata["table_name"] == table_name]
        if meta.empty:
            print(" No metadata found. Skipped.")
            continue

        species_label = meta["species"].values[0]
        condition_label = meta["condition"].values[0]

        query = f"SELECT * FROM {table_name}"
        for chunk in pd.read_sql_query(query, conn, chunksize=batch_size):

            true_species.extend([species_label] * len(chunk))
            true_condition.extend([condition_label] * len(chunk))

            X_encoded = encoder.transform(chunk)
            X_scaled = scaler.transform(X_encoded.toarray())

            score_species_list.append(model_species.predict_proba(X_scaled))
            score_condition_list.append(model_condition.predict_proba(X_scaled))

    conn.close()

    # Stack & fix NaN
    score_species = fix_nan_scores(np.vstack(score_species_list), "Species")
    score_condition = fix_nan_scores(np.vstack(score_condition_list), "Condition")

    species_classes = sorted(list(set(true_species)))
    condition_classes = sorted(list(set(true_condition)))

    y_species_num = pd.Categorical(true_species, categories=species_classes).codes
    y_condition_num = pd.Categorical(true_condition, categories=condition_classes).codes

    y_pred_species = score_species.argmax(axis=1)
    y_pred_condition = score_condition.argmax(axis=1)

    # ---- Compute metrics ----
    metric_summary = {
        "Species": {
            "Accuracy": accuracy_score(y_species_num, y_pred_species),
            "Precision": precision_score(y_species_num, y_pred_species, average="weighted", zero_division=0),
            "Recall": recall_score(y_species_num, y_pred_species, average="weighted", zero_division=0),
            "F1-score": f1_score(y_species_num, y_pred_species, average="weighted", zero_division=0)
        },
        "Condition": {
            "Accuracy": accuracy_score(y_condition_num, y_pred_condition),
            "Precision": precision_score(y_condition_num, y_pred_condition, average="weighted", zero_division=0),
            "Recall": recall_score(y_condition_num, y_pred_condition, average="weighted", zero_division=0),
            "F1-score": f1_score(y_condition_num, y_pred_condition, average="weighted", zero_division=0)
        }
    }

    # ---- Plot all results ----
    plot_metric_summary_heatmap(metric_summary,
                                "Performance Summary",
                                "metric_summary_heatmap.png")

    plot_metric_bar(metric_summary["Species"],
                    "Metric Performance – Species",
                    "metric_bar_species.png")

    plot_metric_bar(metric_summary["Condition"],
                    "Metric Performance – Condition",
                    "metric_bar_condition.png")

    # Confusion Matrices
    cm_species = confusion_matrix(y_species_num, y_pred_species)
    plot_confusion_matrix(cm_species, species_classes, "Confusion Matrix – Species", "cm_species.png")
    plot_confusion_matrix(cm_species, species_classes, "Normalized CM – Species", "cm_species_norm.png", normalize=True)

    cm_condition = confusion_matrix(y_condition_num, y_pred_condition)
    plot_confusion_matrix(cm_condition, condition_classes, "Confusion Matrix – Condition", "cm_condition.png")
    plot_confusion_matrix(cm_condition, condition_classes, "Normalized CM – Condition", "cm_condition_norm.png", normalize=True)

    # PR Curves (species will work, condition is skipped if invalid)
    plot_multiclass_pr(y_species_num, score_species, species_classes,
                       "Precision–Recall – Species", "pr_species.png")

    plot_multiclass_pr(y_condition_num, score_condition, condition_classes,
                       "Precision–Recall – Condition", "pr_condition.png")

    # Distribution
    plot_class_distribution(true_species, species_classes,
                            "Class Distribution – Species", "dist_species.png")

    plot_class_distribution(true_condition, condition_classes,
                            "Class Distribution – Condition", "dist_condition.png")

    # Per-class detailed metrics
    plot_class_metric_heatmap(y_species_num, y_pred_species, species_classes,
                              "Per-Class Metrics – Species", "species_class_metrics.png")

    plot_class_metric_heatmap(y_condition_num, y_pred_condition, condition_classes,
                              "Per-Class Metrics – Condition", "condition_class_metrics.png")

    # Final PDF report
    create_pdf_report("funvisual_report.pdf")

    print("\n Evaluation completed successfully!")

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Memory-safe evaluation visualization")
    parser.add_argument("--db_file", required=True)
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()

    evaluate(args.db_file, args.metadata_file, args.model_file, args.batch_size)

if __name__ == "__main__":
    main()

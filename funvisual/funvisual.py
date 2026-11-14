#!/usr/bin/env python3
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support
)

from matplotlib.backends.backend_pdf import PdfPages
import os
import sys

# shared feature construction (../funml/features.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "funml"))
import features as F  # noqa: E402

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
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype("float"), row_sums, out=np.zeros(cm.shape), where=row_sums > 0)
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
    if n_classes < 2 or y_scores.shape[1] != n_classes:
        print(f" Skipping PR plot '{title}': {n_classes} classes, {y_scores.shape[1]} score columns.")
        return {}
    # one-versus-rest indicator matrix (label_binarize returns a single column for two classes)
    y_bin = np.eye(n_classes, dtype=int)[np.asarray(y_true_num)]
    ap_values = {}
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            print(f" PR curve for '{class_list[i]}' skipped: class not present in the evaluated genomes.")
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_bin[:, i], y_scores[:, i])
        ap_values[class_list[i]] = ap
        plt.plot(recall, precision, label=f"{class_list[i]} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" PR Curve saved: {filename}")
    return ap_values

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

    sns.barplot(x=names, y=values, hue=names, palette="Set2", legend=False)
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
def evaluate(db_file, metadata_file, model_file, split="train", output_dir="."):
    db_file, metadata_file, model_file = map(os.path.abspath, (db_file, metadata_file, model_file))
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    print(" Loading trained model...")
    bundle = F.load_model(model_file)
    species_classes, condition_classes = bundle["species_classes"], bundle["condition_classes"]

    meta = F.read_metadata(metadata_file)
    if split != "all" and "split" in meta.columns:
        tables = list(meta.index[meta["split"].str.lower() == split])
    else:
        tables = list(meta.index)
    if not tables:
        raise ValueError(f"No tables with split '{split}' in {metadata_file}")

    db = F.decompress_db(db_file, ".")
    conn = sqlite3.connect(db)
    y_species, y_condition, p_species, p_condition = [], [], [], []
    for table_name in tables:
        print(f" Evaluating: {table_name}")
        df = F.load_table(conn, table_name)
        ps, pc = F.predict_table(bundle, df)
        p_species.append(ps)
        p_condition.append(pc)
        y_species += [species_classes.index(meta.loc[table_name, "species"])] * len(df)
        y_condition += [condition_classes.index(meta.loc[table_name, "condition"])] * len(df)
    conn.close()

    score_species, score_condition = np.vstack(p_species), np.vstack(p_condition)
    y_species_num, y_condition_num = np.array(y_species), np.array(y_condition)
    y_pred_species = score_species.argmax(axis=1)
    y_pred_condition = score_condition.argmax(axis=1)

    def summary(y, yhat):
        return {"Accuracy": accuracy_score(y, yhat),
                "Precision": precision_score(y, yhat, average="weighted", zero_division=0),
                "Recall": recall_score(y, yhat, average="weighted", zero_division=0),
                "F1-score": f1_score(y, yhat, average="weighted", zero_division=0)}
    metric_summary = {"Species": summary(y_species_num, y_pred_species),
                      "Condition": summary(y_condition_num, y_pred_condition)}

    plot_metric_summary_heatmap(metric_summary, f"Performance Summary ({split} genomes)", "metric_summary_heatmap.png")
    plot_metric_bar(metric_summary["Species"], "Metric Performance - Species", "metric_bar_species.png")
    plot_metric_bar(metric_summary["Condition"], "Metric Performance - Condition", "metric_bar_condition.png")

    cm_species = confusion_matrix(y_species_num, y_pred_species, labels=range(len(species_classes)))
    plot_confusion_matrix(cm_species, species_classes, "Confusion Matrix - Species", "cm_species.png")
    plot_confusion_matrix(cm_species, species_classes, "Normalized CM - Species", "cm_species_norm.png", normalize=True)
    cm_condition = confusion_matrix(y_condition_num, y_pred_condition, labels=range(len(condition_classes)))
    plot_confusion_matrix(cm_condition, condition_classes, "Confusion Matrix - Condition", "cm_condition.png")
    plot_confusion_matrix(cm_condition, condition_classes, "Normalized CM - Condition", "cm_condition_norm.png", normalize=True)

    ap_species = plot_multiclass_pr(y_species_num, score_species, species_classes,
                                    "Precision-Recall - Species", "pr_species.png")
    ap_condition = plot_multiclass_pr(y_condition_num, score_condition, condition_classes,
                                      "Precision-Recall - Condition", "pr_condition.png")

    plot_class_distribution([species_classes[i] for i in y_species_num], species_classes,
                            "Class Distribution - Species", "dist_species.png")
    plot_class_distribution([condition_classes[i] for i in y_condition_num], condition_classes,
                            "Class Distribution - Condition", "dist_condition.png")
    plot_class_metric_heatmap(y_species_num, y_pred_species, species_classes,
                              "Per-Class Metrics - Species", "species_class_metrics.png")
    plot_class_metric_heatmap(y_condition_num, y_pred_condition, condition_classes,
                              "Per-Class Metrics - Condition", "condition_class_metrics.png")

    rows = [{"task": t.lower(), "metric": m, "value": v} for t, d in metric_summary.items() for m, v in d.items()]
    rows += [{"task": "species", "metric": f"AP {c}", "value": v} for c, v in ap_species.items()]
    rows += [{"task": "condition", "metric": f"AP {c}", "value": v} for c, v in ap_condition.items()]
    pd.DataFrame(rows).to_csv("funvisual_metrics.tsv", sep="\t", index=False)

    create_pdf_report("funvisual_report.pdf")
    print("\n Evaluation completed successfully!")
    return metric_summary, ap_species, ap_condition


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluation and visualization of the FunML classifiers")
    parser.add_argument("--db_file", required=True, help="FunLink database (.db or .db.bz2)")
    parser.add_argument("--metadata_file", required=True, help="Metadata TSV (with optional split column)")
    parser.add_argument("--model_file", required=True, help="Model file written by funml.py")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"],
                        help="Genomes to evaluate (default: train, i.e. resubstitution)")
    parser.add_argument("--output_dir", default=".", help="Directory for figures and the PDF report")
    args = parser.parse_args()
    evaluate(args.db_file, args.metadata_file, args.model_file, args.split, args.output_dir)


if __name__ == "__main__":
    main()

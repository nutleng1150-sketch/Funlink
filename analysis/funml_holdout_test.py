#!/usr/bin/env python3
"""
FunML independent hold-out test on the clinical isolate CUPWI001.
2026-11 re-run: 10 reference genomes (FunLink DB v2026-11); only change vs. the
manuscript version is the species label map (new table names use corrected spellings).

The FunML classifiers (species and condition) are trained ONLY on the five
NCBI reference genomes stored in the FunLink database, and the clinical
isolate P. wickerhamii CUPWI001 is used exclusively as an unseen test genome.

Differences from the exploratory funml.py / funvisual.py run (documented in
README.md of this folder):
  * CUPWI001 is excluded from training (previously it was one of the six
    tables the model was fitted on).
  * Encoder and scaler are fitted on the training genomes only.
  * Columns that carry the label or identify the genome (id, isolate, species,
    condition, locus/contig identifiers, genome coordinates) and raw sequence
    strings are removed from the feature set.
  * Training rows are shuffled before incremental learning.

usage:
python funml_holdout_test.py --db_bz2_file <efficient_data_storage.db.bz2> \
       --metadata_file metadata_holdout.tsv --output_dir .
"""

import os
import bz2
import json
import shutil
import zlib
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# -------------------- Fixed configuration (set before testing) --------------------
SEED = 42
N_SEEDS = 10          # repeated runs to show sensitivity to SGD initialisation/shuffling
N_EPOCHS = 50         # 5 epochs did not converge on the training genomes (see README.md)
BATCH_SIZE = 1000
N_BOOTSTRAP = 1000

# Columns never used as features
LABEL_COLS = ["id", "isolate", "species", "condition"]
GENOME_ID_COLS = ["qseqid", "GeneID", "TranscriptID", "Contig", "Start", "Stop", "Strand"]
SEQUENCE_COLS = ["gDNA", "mRNA", "CDS-transcript", "Translation", "Sequence"]
NUMERIC_COLS = ["qstart", "qend", "sstart", "send", "pident", "evalue",
                "Length", "Mass", "Annotation", "Sequence version", "Entry version"]

SPECIES_LABEL = {"prototheca_bovia": "P. bovis", "prototheca_cifferri": "P. ciferrii",
                 "prototheca_bovis": "P. bovis", "prototheca_ciferrii": "P. ciferrii",
                 "prototheca_wickerhamii": "P. wickerhamii"}
CONDITION_LABEL = {"clinical_bovine": "Bovine", "clinical_human": "Human"}


# -------------------- Utility Functions --------------------
def decompress_db(bz2_file, output_dir):
    """Decompress the database if it's compressed."""
    if not bz2_file.endswith(".bz2"):
        return bz2_file
    db_file = os.path.join(output_dir, "efficient_data_storage.db")
    if not os.path.exists(db_file):
        print(f"Decompressing {bz2_file}...")
        with bz2.BZ2File(bz2_file, "rb") as f_in, open(db_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return db_file


def load_table(conn, table_name):
    """Load one genome table and decompress the zlib-compressed columns."""
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: zlib.decompress(x).decode("utf-8") if isinstance(x, bytes) else x)
    return df


def split_features(df):
    """Return (categorical frame, numeric frame) with label/identifier columns removed."""
    drop = [c for c in LABEL_COLS + GENOME_ID_COLS + SEQUENCE_COLS if c in df.columns]
    X = df.drop(columns=drop)
    num = X[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    num["evalue"] = -np.log10(num["evalue"].clip(lower=1e-300))
    cat = X.drop(columns=NUMERIC_COLS).astype(str)
    return cat, num


class Featurizer:
    """OneHotEncoder for annotation fields + StandardScaler for numeric fields (fit on training only)."""

    def fit(self, cat, num):
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.encoder.fit(cat)
        self.scaler.fit(num)
        return self

    def transform(self, cat, num):
        return sparse.hstack([self.encoder.transform(cat),
                              sparse.csr_matrix(self.scaler.transform(num))]).tocsr()


def fix_nan_scores(scores):
    """Correct NaN / zero-sum posterior rows and re-normalise (as in funvisual.py)."""
    scores = np.array(scores, dtype=float)
    bad = ~np.isfinite(scores).all(axis=1)
    scores[bad] = 1.0 / scores.shape[1]
    sums = scores.sum(axis=1, keepdims=True)
    zero = (sums == 0).squeeze()
    scores[zero] = 1.0 / scores.shape[1]
    return scores / scores.sum(axis=1, keepdims=True)


def train_incremental(X, y, n_classes, seed, balanced=False):
    """Incremental averaged SGD (multiclass logistic regression) trained by partial_fit on shuffled batches."""
    model = SGDClassifier(loss="log_loss", average=True, random_state=seed)
    rng = np.random.RandomState(seed)
    classes = np.arange(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = (len(y) / (np.count_nonzero(counts) * np.maximum(counts, 1)))[y] if balanced else np.ones(len(y))
    for _ in range(N_EPOCHS):
        order = rng.permutation(X.shape[0])
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            model.partial_fit(X[idx], y[idx], classes=classes, sample_weight=weights[idx])
    return model


def posterior(model, X, n_classes):
    """Posterior matrix with one column per class id (handles models that saw fewer classes)."""
    out = np.zeros((X.shape[0], n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return fix_nan_scores(out)


def weighted_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def run_split(frames, meta, train_tables, test_table, classes, target, seed, balanced=False):
    """Train on train_tables, test on test_table for one target ('species' or 'condition')."""
    cat_tr = pd.concat([frames[t][0] for t in train_tables], ignore_index=True)
    num_tr = pd.concat([frames[t][1] for t in train_tables], ignore_index=True)
    y_tr = np.concatenate([[classes.index(meta.loc[t, target])] * len(frames[t][0]) for t in train_tables])
    fz = Featurizer().fit(cat_tr, num_tr)
    X_tr = fz.transform(cat_tr, num_tr)
    model = train_incremental(X_tr, y_tr, len(classes), seed, balanced)
    X_te = fz.transform(*frames[test_table])
    return {"model": model, "n_features": X_tr.shape[1],
            "y_train": y_tr, "p_train": posterior(model, X_tr, len(classes)),
            "p_test": posterior(model, X_te, len(classes))}


# -------------------- Figure --------------------
def make_figure(summary, out_base):
    ink, muted = "#0b0b0b", "#898781"
    colors = {"P. bovis": "#2a78d6", "P. ciferrii": "#eb6834", "P. wickerhamii": "#1baf7a",
              "Bovine": "#2a78d6", "Human": "#1baf7a"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [3, 2]})
    panels = [("species", "A  Species"), ("condition", "B  Condition")]
    for ax, (target, title) in zip(axes, panels):
        s = summary[target]
        names = s["class_names"]
        x = np.arange(len(names))
        w = 0.38
        frac = np.array(s["test_fraction_predicted"])
        mean_p = np.array(s["test_mean_posterior"])
        lo, hi = np.array(s["test_mean_posterior_ci95"]).T
        cols = [colors[n] for n in names]
        ax.bar(x - w / 2 - 0.01, frac, w, color=cols, edgecolor="white", linewidth=0.8)
        ax.bar(x + w / 2 + 0.01, mean_p, w, color=cols, edgecolor="white", linewidth=0.8,
               hatch="////", alpha=0.75, yerr=[mean_p - lo, hi - mean_p],
               error_kw={"ecolor": ink, "elinewidth": 0.8, "capsize": 2})
        for xi, f, m in zip(x, frac, mean_p):
            ax.text(xi - w / 2 - 0.01, f + 0.02, f"{f:.2f}", ha="center", va="bottom", fontsize=8, color=ink)
            ax.text(xi + w / 2 + 0.01, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=ink)
        ax.axhline(1.0 / len(names), color=muted, linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontstyle="italic" if target == "species" else "normal")
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=ink)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted class")
    axes[0].set_ylabel("Proportion / probability")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#bdbcb6", edgecolor="white", label="Fraction of feature vectors assigned"),
                        Patch(facecolor="#bdbcb6", edgecolor="white", hatch="////", alpha=0.75,
                              label="Genome-level mean posterior (95% CI)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# -------------------- Main Function --------------------
def main():
    parser = argparse.ArgumentParser(description="Hold-out test of FunML on an unseen genome table.")
    parser.add_argument("--db_bz2_file", required=True, help="Path to FunLink SQLite database (.db or .db.bz2)")
    parser.add_argument("--metadata_file", required=True, help="Metadata TSV with a 'split' column (train/test)")
    parser.add_argument("--output_dir", required=True, help="Directory for result files")
    parser.add_argument("--work_dir", default=None, help="Directory for the decompressed database")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    db_file = decompress_db(args.db_bz2_file, args.work_dir or args.output_dir)
    meta = pd.read_csv(args.metadata_file, sep="\t").set_index("table_name")
    train_tables = list(meta.index[meta["split"] == "train"])
    test_table = meta.index[meta["split"] == "test"][0]

    conn = sqlite3.connect(db_file)
    raw = {t: load_table(conn, t) for t in meta.index}
    conn.close()
    frames = {t: split_features(df) for t, df in raw.items()}

    classes = {"species": sorted(meta.loc[train_tables, "species"].unique()),
               "condition": sorted(meta.loc[train_tables, "condition"].unique())}
    pretty = {"species": SPECIES_LABEL, "condition": CONDITION_LABEL}

    dataset_rows = [{"table_name": t, "isolate": meta.loc[t, "isolate"], "species": meta.loc[t, "species"],
                     "condition": meta.loc[t, "condition"], "split": meta.loc[t, "split"],
                     "n_feature_vectors": len(raw[t]),
                     "n_distinct_best_hit_proteins": int(raw[t]["Unique ID"].nunique()),
                     "median_pident": round(float(raw[t]["pident"].median()), 2),
                     "fraction_evalue_below_1e-5": round(float((raw[t]["evalue"] < 1e-5).mean()), 4)} for t in meta.index]
    all_hits = pd.concat([raw[t]["Unique ID"] for t in meta.index])
    print("Distinct best-hit reference proteins across all genomes:", all_hits.nunique())
    pd.DataFrame(dataset_rows).to_csv(os.path.join(args.output_dir, "dataset_summary.tsv"), sep="\t", index=False)

    summary = {"config": {"seed": SEED, "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
                          "classifier": "SGDClassifier(loss='log_loss', average=True)", "n_bootstrap": N_BOOTSTRAP,
                          "train_tables": train_tables, "test_table": test_table,
                          "n_train_vectors": int(sum(len(raw[t]) for t in train_tables)),
                          "n_test_vectors": int(len(raw[test_table])),
                          "n_distinct_best_hit_proteins_all_genomes": int(all_hits.nunique())}}
    metric_rows, seed_rows, logo_rows, control_rows = [], [], [], []
    pred = raw[test_table][["qseqid", "Product", "Unique ID", "Organism", "pident", "evalue"]].copy()

    for target in ["species", "condition"]:
        cls = classes[target]
        names = [pretty[target][c] for c in cls]
        true_idx = cls.index(meta.loc[test_table, target])

        # ---- primary run (SEED) ----
        r = run_split(frames, meta, train_tables, test_table, cls, target, SEED)
        yhat_tr = r["p_train"].argmax(axis=1)
        m_tr = weighted_metrics(r["y_train"], yhat_tr)
        p_te = r["p_test"]
        yhat_te = p_te.argmax(axis=1)
        y_te = np.full(len(yhat_te), true_idx)
        m_te = weighted_metrics(y_te, yhat_te)
        mean_p = p_te.mean(axis=0)

        # bootstrap over feature vectors (within-genome sampling uncertainty only)
        rng = np.random.RandomState(SEED)
        boot_mean, boot_acc = [], []
        for _ in range(N_BOOTSTRAP):
            idx = rng.randint(0, len(p_te), len(p_te))
            boot_mean.append(p_te[idx].mean(axis=0))
            boot_acc.append((yhat_te[idx] == true_idx).mean())
        ci_mean = np.percentile(np.array(boot_mean), [2.5, 97.5], axis=0).T
        ci_acc = np.percentile(boot_acc, [2.5, 97.5])

        for split, m, n in [(f"training ({len(train_tables)} reference genomes, resubstitution)", m_tr, len(yhat_tr)),
                            ("independent test (CUPWI001)", m_te, len(yhat_te))]:
            metric_rows.append({"task": target, "dataset": split, "n_feature_vectors": n,
                                **{k: round(v, 4) for k, v in m.items()}})

        cm_tr = confusion_matrix(r["y_train"], yhat_tr, labels=range(len(cls)))
        pd.DataFrame(cm_tr, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]) \
            .to_csv(os.path.join(args.output_dir, f"confusion_matrix_training_{target}.tsv"), sep="\t")

        # ---- controls: majority-class baseline and class-balanced training ----
        prior = np.bincount(r["y_train"], minlength=len(cls)) / len(r["y_train"])
        rb = run_split(frames, meta, train_tables, test_table, cls, target, SEED, balanced=True)
        mb = rb["p_test"].mean(axis=0)
        control_rows.append({"task": target, "model": "majority-class baseline (always predicts %s)" % names[int(prior.argmax())],
                             "train_accuracy": round(float(prior.max()), 4),
                             "test_accuracy": round(float(prior.argmax() == true_idx), 4),
                             "test_mean_posterior_true_class": round(float(prior[true_idx]), 4),
                             "genome_level_MAP_call": names[int(prior.argmax())]})
        control_rows.append({"task": target, "model": "FunML (primary, unweighted)",
                             "train_accuracy": round(m_tr["accuracy"], 4), "test_accuracy": round(m_te["accuracy"], 4),
                             "test_mean_posterior_true_class": round(float(mean_p[true_idx]), 4),
                             "genome_level_MAP_call": names[int(mean_p.argmax())]})
        control_rows.append({"task": target, "model": "FunML (class-balanced sample weights)",
                             "train_accuracy": round(accuracy_score(rb["y_train"], rb["p_train"].argmax(axis=1)), 4),
                             "test_accuracy": round(float((rb["p_test"].argmax(axis=1) == true_idx).mean()), 4),
                             "test_mean_posterior_true_class": round(float(mb[true_idx]), 4),
                             "genome_level_MAP_call": names[int(mb.argmax())]})

        counts = np.bincount(yhat_te, minlength=len(cls))
        summary[target] = {
            "training_class_prior": prior.tolist(),
            "balanced_test_mean_posterior": mb.tolist(),
            "balanced_test_accuracy": float((rb["p_test"].argmax(axis=1) == true_idx).mean()),
            "balanced_genome_level_MAP_call": names[int(mb.argmax())],
            "class_names": names, "true_class": names[true_idx], "n_features": int(r["n_features"]),
            "training_metrics": m_tr, "test_metrics": m_te,
            "test_accuracy_ci95": [float(ci_acc[0]), float(ci_acc[1])],
            "test_predicted_counts": counts.tolist(),
            "test_fraction_predicted": (counts / counts.sum()).tolist(),
            "test_mean_posterior": mean_p.tolist(),
            "test_mean_posterior_ci95": ci_mean.tolist(),
            "genome_level_MAP_call": names[int(mean_p.argmax())],
            "genome_level_call_correct": bool(mean_p.argmax() == true_idx)}

        pred[f"{target}_predicted"] = [names[i] for i in yhat_te]
        for j, n in enumerate(names):
            pred[f"{target}_posterior_{n}"] = np.round(p_te[:, j], 4)

        # ---- sensitivity to random seed ----
        for s in range(SEED, SEED + N_SEEDS):
            rs = r if s == SEED else run_split(frames, meta, train_tables, test_table, cls, target, s)
            mp = rs["p_test"].mean(axis=0)
            seed_rows.append({"task": target, "seed": s,
                              "train_accuracy": round(accuracy_score(rs["y_train"], rs["p_train"].argmax(axis=1)), 4),
                              "test_accuracy": round(float((rs["p_test"].argmax(axis=1) == true_idx).mean()), 4),
                              "test_mean_posterior_true_class": round(float(mp[true_idx]), 4),
                              "genome_level_MAP_call": names[int(mp.argmax())]})

        # ---- leave-one-genome-out within the training genomes ----
        # Only possible for classes that keep >= 1 training genome when one is removed.
        for held in train_tables:
            rest = [t for t in train_tables if t != held]
            if meta.loc[held, target] not in set(meta.loc[rest, target]):
                logo_rows.append({"task": target, "held_out_genome": meta.loc[held, "isolate"],
                                  "true_class": pretty[target][meta.loc[held, target]],
                                  "n_feature_vectors": len(raw[held]), "accuracy": "",
                                  "mean_posterior_true_class": "", "genome_level_MAP_call": "",
                                  "note": "not evaluable: only genome of its class"})
                continue
            rl = run_split(frames, meta, rest, held, cls, target, SEED)
            ti = cls.index(meta.loc[held, target])
            mp = rl["p_test"].mean(axis=0)
            logo_rows.append({"task": target, "held_out_genome": meta.loc[held, "isolate"],
                              "true_class": names[ti], "n_feature_vectors": len(raw[held]),
                              "accuracy": round(float((rl["p_test"].argmax(axis=1) == ti).mean()), 4),
                              "mean_posterior_true_class": round(float(mp[ti]), 4),
                              "genome_level_MAP_call": names[int(mp.argmax())], "note": ""})
        print(f"Finished task: {target}")

    seed_df = pd.DataFrame(seed_rows)
    for target in ["species", "condition"]:
        d = seed_df[seed_df["task"] == target]
        summary[target]["seed_sensitivity"] = {
            "test_accuracy_mean": float(d["test_accuracy"].mean()), "test_accuracy_sd": float(d["test_accuracy"].std()),
            "test_accuracy_min": float(d["test_accuracy"].min()), "test_accuracy_max": float(d["test_accuracy"].max()),
            "train_accuracy_mean": float(d["train_accuracy"].mean()), "train_accuracy_sd": float(d["train_accuracy"].std()),
            "n_correct_genome_calls": int((d["genome_level_MAP_call"] == summary[target]["true_class"]).sum()),
            "n_seeds": int(len(d))}

    out = args.output_dir
    pd.DataFrame(metric_rows).to_csv(os.path.join(out, "metrics_training_vs_test.tsv"), sep="\t", index=False)
    seed_df.to_csv(os.path.join(out, "seed_sensitivity.tsv"), sep="\t", index=False)
    pd.DataFrame(control_rows).to_csv(os.path.join(out, "controls_baseline_and_balanced.tsv"), sep="\t", index=False)
    pd.DataFrame(logo_rows).to_csv(os.path.join(out, "leave_one_genome_out.tsv"), sep="\t", index=False)
    pred.to_csv(os.path.join(out, "CUPWI001_per_protein_predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(out, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_figure(summary, os.path.join(out, "Figure5_CUPWI001_holdout_test"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

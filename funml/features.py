#!/usr/bin/env python3
"""
Shared FunML code: database access, feature construction, incremental training and posteriors.
Used by funml.py (training), funvisual.py (evaluation) and predict.py (genome-level prediction),
so that all three build features in exactly the same way.

Features: fields that carry the class label or identify the source genome, and raw sequences,
are removed; the remaining annotation fields are one-hot encoded and the numeric fields
(alignment coordinates, percentage identity, -log10 E-value, protein length and mass,
annotation score, entry versions) are standardized. Encoder and scaler are fitted on the
training genomes only.
"""
import os
import bz2
import shutil
import zlib
import sqlite3
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Training defaults (the settings used in the manuscript)
SEED = 42
N_EPOCHS = 50
BATCH_SIZE = 1000

LABEL_COLS = ["id", "isolate", "species", "condition"]
GENOME_ID_COLS = ["qseqid", "GeneID", "TranscriptID", "Contig", "Start", "Stop", "Strand"]
SEQUENCE_COLS = ["gDNA", "mRNA", "CDS-transcript", "Translation", "Sequence"]
NUMERIC_COLS = ["qstart", "qend", "sstart", "send", "pident", "evalue",
                "Length", "Mass", "Annotation", "Sequence version", "Entry version"]


def decompress_db(db_path, work_dir):
    """Return a path to an uncompressed SQLite file (decompressing a .bz2 database into work_dir)."""
    if not db_path.endswith(".bz2"):
        return db_path
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, os.path.basename(db_path)[:-4])
    if not os.path.exists(out):
        print(f"Decompressing {db_path} ...")
        with bz2.BZ2File(db_path, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out


def load_table(conn, table_name):
    """Load one genome table and decompress its zlib-compressed text columns."""
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: zlib.decompress(x).decode("utf-8") if isinstance(x, bytes) else x)
    return df


def split_features(df):
    """Return (categorical frame, numeric frame) with label, identifier and sequence columns removed."""
    drop = [c for c in LABEL_COLS + GENOME_ID_COLS + SEQUENCE_COLS if c in df.columns]
    X = df.drop(columns=drop)
    num = X.reindex(columns=NUMERIC_COLS).apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    num["evalue"] = -np.log10(num["evalue"].clip(lower=1e-300))
    cat = X.drop(columns=[c for c in NUMERIC_COLS if c in X.columns]).astype(str)
    return cat, num


class Featurizer:
    """OneHotEncoder for annotation fields + StandardScaler for numeric fields (fit on training genomes only)."""

    def fit(self, cat, num):
        self.cat_columns = list(cat.columns)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.encoder.fit(cat)
        self.scaler.fit(num)
        return self

    def transform(self, cat, num):
        cat = cat.reindex(columns=self.cat_columns, fill_value="no information")
        return sparse.hstack([self.encoder.transform(cat),
                              sparse.csr_matrix(self.scaler.transform(num))]).tocsr()


def fix_nan_scores(scores):
    """Replace NaN/inf or zero-sum posterior rows by the uniform distribution and re-normalise each row."""
    scores = np.array(scores, dtype=float)
    bad = ~np.isfinite(scores).all(axis=1)
    scores[bad] = 1.0 / scores.shape[1]
    sums = scores.sum(axis=1, keepdims=True)
    zero = (sums == 0).squeeze(axis=1)
    scores[zero] = 1.0 / scores.shape[1]
    return scores / scores.sum(axis=1, keepdims=True)


def train_incremental(X, y, n_classes, seed=SEED, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, balanced=False):
    """Averaged SGD logistic regression (one-versus-rest) trained by partial_fit on shuffled mini-batches."""
    model = SGDClassifier(loss="log_loss", average=True, random_state=seed)
    rng = np.random.RandomState(seed)
    classes = np.arange(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = (len(y) / (np.count_nonzero(counts) * np.maximum(counts, 1)))[y] if balanced else np.ones(len(y))
    for _ in range(n_epochs):
        order = rng.permutation(X.shape[0])
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            model.partial_fit(X[idx], y[idx], classes=classes, sample_weight=weights[idx])
    return model


def posterior(model, X, n_classes):
    """Posterior matrix with one column per class (handles models that saw fewer classes)."""
    out = np.zeros((X.shape[0], n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return fix_nan_scores(out)


def read_metadata(metadata_file):
    """Metadata TSV with columns table_name, species, condition, isolate and an optional split (train/test)."""
    meta = pd.read_csv(metadata_file, sep="\t", dtype=str)
    missing = {"table_name", "species", "condition"} - set(meta.columns)
    if missing:
        raise ValueError(f"metadata file lacks columns: {sorted(missing)}")
    return meta.set_index("table_name")


def load_model(model_file):
    import joblib
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or "featurizer" not in bundle:
        raise ValueError("Model file was not produced by this FunML version; please re-train with funml.py.")
    return bundle


def predict_table(bundle, df):
    """Posterior matrices (species, condition) for one genome table."""
    X = bundle["featurizer"].transform(*split_features(df))
    return (posterior(bundle["model_species"], X, len(bundle["species_classes"])),
            posterior(bundle["model_condition"], X, len(bundle["condition_classes"])))

#!/usr/bin/env python3
"""
FunML: train the species and condition classifiers on genome tables stored by FunLink.

Each genome table is read from the SQLite database (text fields are zlib-decompressed), converted
into feature vectors (see features.py), and the two classifiers are trained incrementally with
partial_fit on shuffled mini-batches (SGDClassifier, logistic loss, one-versus-rest, weight averaging).
Only tables marked 'train' in the metadata 'split' column are used (all tables listed, if there is no
'split' column), so genomes reserved for testing never influence the encoder, scaler or model.

usage:
python funml.py --db_bz2_file efficient_data_storage.db.bz2 --metadata_file metadata.tsv \
                --output_dir work --model_output model.pkl [--epochs 50 --batch_size 1000 --seed 42]
"""
import os
import sys
import argparse
import sqlite3
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402


def train(db_file, metadata_file, model_output, epochs=F.N_EPOCHS, batch_size=F.BATCH_SIZE, seed=F.SEED):
    meta = F.read_metadata(metadata_file)
    if "split" in meta.columns:
        train_tables = list(meta.index[meta["split"].str.lower() == "train"])
    else:
        train_tables = list(meta.index)
    if not train_tables:
        raise ValueError("No training tables found in the metadata file.")

    conn = sqlite3.connect(db_file)
    frames = {}
    for t in train_tables:
        df = F.load_table(conn, t)
        frames[t] = F.split_features(df)
        print(f"Loaded {t}: {len(df)} feature vectors")
    conn.close()

    cat = pd.concat([frames[t][0] for t in train_tables], ignore_index=True)
    num = pd.concat([frames[t][1] for t in train_tables], ignore_index=True)
    featurizer = F.Featurizer().fit(cat, num)
    X = featurizer.transform(cat, num)
    print(f"Training matrix: {X.shape[0]} feature vectors x {X.shape[1]} features")

    bundle = {"featurizer": featurizer,
              "config": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                         "train_tables": train_tables, "n_features": int(X.shape[1])}}
    for target in ["species", "condition"]:
        classes = sorted(meta.loc[train_tables, target].unique())
        y = np.concatenate([[classes.index(meta.loc[t, target])] * len(frames[t][0]) for t in train_tables])
        print(f"Training {target} classifier ({len(classes)} classes: {', '.join(classes)}) ...")
        bundle[f"model_{target}"] = F.train_incremental(X, y, len(classes), seed, epochs, batch_size)
        bundle[f"{target}_classes"] = classes

    joblib.dump(bundle, model_output)
    print(f"Model saved to {model_output}")


def main():
    p = argparse.ArgumentParser(description="Incremental training of the FunML species and condition classifiers")
    p.add_argument("--db_bz2_file", required=True, help="FunLink database (.db or .db.bz2)")
    p.add_argument("--metadata_file", required=True, help="TSV: table_name, species, condition[, isolate, split]")
    p.add_argument("--output_dir", required=True, help="Directory for the decompressed database")
    p.add_argument("--model_output", required=True, help="Output path for the model file (.pkl)")
    p.add_argument("--epochs", type=int, default=F.N_EPOCHS, help="Passes over the training data (default 50)")
    p.add_argument("--batch_size", type=int, default=F.BATCH_SIZE, help="Mini-batch size (default 1000)")
    p.add_argument("--seed", type=int, default=F.SEED, help="Random seed (default 42)")
    a = p.parse_args()
    db = F.decompress_db(a.db_bz2_file, a.output_dir)
    train(db, a.metadata_file, a.model_output, a.epochs, a.batch_size, a.seed)


if __name__ == "__main__":
    main()

# Funlink
FunLink links the output of a genome annotation pipeline (currently Funannotate) to UniProtKB and stores
each genome as one table in a compressed SQLite database, indexed by species, condition and isolate.
FunML trains species and condition classifiers on the stored genomes, and FunVisual evaluates them.

Requirements: Python >= 3.8, pandas, numpy, scipy, scikit-learn (>= 1.1), matplotlib, seaborn, requests,
joblib, and BLAST+ for the BLASTP search.

## Funlink
usage (`funlink/run.py`):
--protein_fasta: reference protein file (UniProtKB proteins in this version)
--output_db: path to the BLASTP database
--query_fasta: predicted proteins of the genome (Funannotate `*.proteins.fa`)
--annotation_file: Funannotate annotation table (`*.annotations.txt`)

## Funml
Incremental machine learning (SGDClassifier, logistic loss, one-versus-rest, weight averaging) on the
genome tables stored in the database. Label fields (species, condition, isolate), genome identifiers
(locus, contig, coordinates) and raw sequences are excluded from the features; annotation fields are
one-hot encoded and numeric fields standardized, with the encoder and scaler fitted on the training
genomes only. Defaults: 50 passes over the data, mini-batches of 1,000 feature vectors, seed 42.

```
python funml/funml.py --db_bz2_file efficient_data_storage.db.bz2 --metadata_file metadata.tsv \
                      --output_dir work --model_output model.pkl
```

Metadata file (tab-separated): `table_name`, `species`, `condition`, `isolate` and an optional `split`
column (`train` / `test`). Only `train` tables are used for training (all tables if there is no `split`
column). See `analysis/metadata_holdout.tsv` for an example.

## Funvisual
Evaluation and visualization (confusion matrices, precision-recall curves, class distributions, metric
heatmaps, PDF report and `funvisual_metrics.tsv`), and genome-level prediction with an HTML report.

```
python funvisual/funvisual.py --db_file efficient_data_storage.db.bz2 --metadata_file metadata.tsv \
                              --model_file model.pkl --split train --output_dir results_train
python funvisual/predict.py --db_file work/efficient_data_storage.db --model_file model.pkl \
                            --table_name <species_condition_isolate> --html_out prediction.html
```

`--split train` gives training-set (resubstitution) metrics, `--split test` evaluates the held-out
genomes, `--split all` evaluates every table in the metadata file. `predict.py` averages the
per-feature-vector posteriors of one genome and reports the class with the highest mean (MAP).

## Analysis
`analysis/` holds the hold-out test used for the manuscript (Table 2, Figure 5); see its README.

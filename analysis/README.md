# Analysis scripts for the FunLink/FunML Prototheca manuscript

| File | Purpose |
|---|---|
| `funml_holdout_test.py` | Hold-out test: trains the species and condition classifiers on the reference genomes only and tests the clinical isolate CUPWI001 (Table 2, Figure 5); also majority-class baseline, class-balanced model, leave-one-genome-out and 10-seed sensitivity |
| `clean_fig34.py` | Training-set (resubstitution) metrics and precision–recall curves with the same model, seed 42 (Figures 3 and 4) |
| `figure2_output_size.py` | Output size of Funannotate versus FunLink per genome (Figure 2) |
| `metadata_holdout.tsv` | Genome tables, labels and train/test split (10 reference genomes for training, CUPWI001 for testing) |

Condition labels follow the host association of each species (P. wickerhamii = human; P. bovis and
P. ciferrii = bovine), as described in the manuscript.

```
python funml_holdout_test.py --db_bz2_file efficient_data_storage.db.bz2 --metadata_file metadata_holdout.tsv --output_dir results --work_dir tmp
python clean_fig34.py tmp/efficient_data_storage.db metadata_holdout.tsv results
python figure2_output_size.py
```

Tested with Python 3.13, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, matplotlib and seaborn.
The FunLink database (`efficient_data_storage.db.bz2`) is built with the `funlink` module from
Funannotate output and a UniProtKB reference protein set.

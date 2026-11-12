# Figures 3-4 recomputed on the reference genomes only, with the same leak-free model as the hold-out test (seed 42).
import sys, sqlite3, json, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import funml_holdout_test as H
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
db, meta_f, out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = pd.read_csv(meta_f, sep="\t").set_index("table_name")
train = list(meta.index[meta["split"] == "train"]); test = meta.index[meta["split"] == "test"][0]
conn = sqlite3.connect(db); raw = {t: H.load_table(conn, t) for t in meta.index}; conn.close()
frames = {t: H.split_features(df) for t, df in raw.items()}
res = {}
for target in ["species", "condition"]:
    cls = sorted(meta.loc[train, target].unique())
    r = H.run_split(frames, meta, train, test, cls, target, H.SEED)
    y, p = r["y_train"], r["p_train"]; m = H.weighted_metrics(y, p.argmax(1))
    ap = {H.SPECIES_LABEL.get(c, H.CONDITION_LABEL.get(c, c)): float(average_precision_score(y == i, p[:, i])) for i, c in enumerate(cls)}
    prev = {H.SPECIES_LABEL.get(c, H.CONDITION_LABEL.get(c, c)): float((y == i).mean()) for i, c in enumerate(cls)}
    res[target] = {"metrics": m, "AP": ap, "prevalence": prev}
    if target == "species":
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for i, c in enumerate(cls):
            pr, rc, _ = precision_recall_curve(y == i, p[:, i]); n = H.SPECIES_LABEL[c]
            ax.plot(rc, pr, label=f"{n} (AP = {ap[n]:.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.03)
        ax.set_title(f"Precision–recall, species (training set, {len(train)} reference genomes)", fontsize=10)
        leg = ax.legend(loc="lower left", frameon=False)
        for t in leg.get_texts(): t.set_fontstyle("italic")
        fig.tight_layout(); fig.savefig(out + "/Figure_4_clean.png", dpi=300); fig.savefig(out + "/Figure_4_clean.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
df = pd.DataFrame({"Species": [res["species"]["metrics"][k] for k in ["accuracy","precision_weighted","recall_weighted","f1_weighted"]],
                   "Condition": [res["condition"]["metrics"][k] for k in ["accuracy","precision_weighted","recall_weighted","f1_weighted"]]},
                  index=["Accuracy", "Precision", "Recall", "F1-score"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "Score"}, annot_kws={"size": 11})
ax.set_title(f"Performance summary (training set, {len(train)} reference genomes)", fontsize=12); ax.set_xlabel("Task"); ax.set_ylabel("Metric")
fig.tight_layout(); fig.savefig(out + "/Figure_3_clean.png", dpi=300); plt.close(fig)
json.dump(res, open(out + "/clean_fig34.json", "w"), indent=1); print(json.dumps(res, indent=1))

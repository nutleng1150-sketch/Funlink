# Figure 2 (v5): size of the complete Funannotate output (annotate_results folder) versus the
# FunLink output (the genome's table alone in an SQLite database, bz2-compressed as stored by FunLink).
# Sizes measured 2026-11-12 from the Funannotate work-folder archives and FunLink DB v2026-11 (MB = 10^6 bytes).
# P. ciferrii SAG 2063 is not shown: its complete Funannotate output folder was not retained.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rows = [  # label, species, Funannotate annotate_results (MB), FunLink table bz2 (MB), FunLink records
    ("ATCC 16529", "P. wickerhamii", 142.6, 6.7, 4007),
    ("UVCC 0001", "P. wickerhamii", 189.2, 10.5, 5769),
    ("InPu22FZ", "P. wickerhamii", 199.0, 11.2, 6076),
    ("PwS1", "P. wickerhamii", 189.2, 10.8, 5756),
    ("Pw26", "P. wickerhamii", 190.1, 10.7, 5734),
    ("CUPWI001", "P. wickerhamii", 185.3, 10.6, 5962),
    ("SAG 2021", "P. bovis", 254.8, 15.0, 6931),
    ("P18", "P. bovis", 256.1, 15.3, 6171),
    ("SH08", "P. bovis", 288.6, 17.4, 6934),
    ("SH13", "P. ciferrii", 296.5, 17.6, 6957),
]
df = pd.DataFrame(rows, columns=["strain", "species", "funannotate_MB", "funlink_MB", "records"])
df["ratio"] = df.funannotate_MB / df.funlink_MB
df.to_csv(__file__.replace(".py", ".tsv"), sep="\t", index=False)

ink, muted = "#0b0b0b", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = np.arange(len(df))[::-1]
h = 0.38
ax.barh(y + h / 2, df.funannotate_MB, h, color="#2a78d6", label="Funannotate output (annotate_results folder)")
ax.barh(y - h / 2, df.funlink_MB, h, color="#eb6834", label="FunLink output (genome table, SQLite + bz2)")
for yi, a, b in zip(y, df.funannotate_MB, df.funlink_MB):
    ax.text(a + 3, yi + h / 2, f"{a:.0f}", va="center", fontsize=8, color=ink)
    ax.text(b + 3, yi - h / 2, f"{b:.1f}", va="center", fontsize=8, color=ink)
ax.set_yticks(y)
ax.set_yticklabels([""] * len(df))
for yi, (sp, st) in zip(y, zip(df.species, df.strain)):
    ax.text(-4, yi, st, ha="right", va="center", fontsize=9, color=ink)
    ax.text(-4, yi - 0.36, sp, ha="right", va="center", fontsize=7.5, color=muted, fontstyle="italic")
ax.set_xlim(0, 330)
ax.set_xlabel("Output size (MB)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False, fontsize=8)
fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
out = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0]
fig.savefig(out + "/Figure_2_output_size.png", dpi=300)
fig.savefig(out + "/Figure_2_output_size.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print(df.to_string(index=False))
print("ratio range %.0f-%.0f" % (df.ratio.min(), df.ratio.max()))

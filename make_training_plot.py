import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\dilya\Downloads\results.csv")
df.columns = [c.strip() for c in df.columns]

epoch = df["epoch"]
best = 30  

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("YOLOv11 Training Curves", fontsize=15, fontweight="bold")

panels = [
    (axes[0, 0], "train/box_loss",        "Train Box Loss",     "#1f77b4"),
    (axes[0, 1], "train/cls_loss",        "Train Class Loss",   "#ff7f0e"),
    (axes[1, 0], "metrics/mAP50(B)",      "Val mAP@0.5",        "#9467bd"),
    (axes[1, 1], "metrics/mAP50-95(B)",   "Val mAP@0.5:0.95",   "#d62728"),
]

for ax, col, title, color in panels:
    ax.plot(epoch, df[col], color=color, linewidth=1.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_xlim(1, epoch.max())
    ax.grid(True, alpha=0.3)
    # mark best epoch on the two validation panels
    if col.startswith("metrics"):
        ax.axvline(best, color="green", linestyle="--", linewidth=1.2, alpha=0.8)
        yval = df.loc[df["epoch"] == best, col].values[0]
        ax.scatter([best], [yval], color="green", zorder=5, s=40)
        ax.annotate(f"best (epoch {best})", xy=(best, yval),
                    xytext=(6, -12), textcoords="offset points",
                    fontsize=9, color="green")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = r"C:\Users\dilya\Desktop\Computer_Vision\reports\yolo_training_curves_clean.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

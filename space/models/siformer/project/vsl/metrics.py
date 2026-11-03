from pathlib import Path

import numpy as np

from .common import write_json, write_csv


def export_metrics(rows, labels, out):
    """Failures are abstentions: included as FN and in the Top-1 denominator."""
    out = Path(out)
    names = [name for name, i in sorted(labels.items(), key=lambda pair: pair[1])]
    cm = np.zeros((100, 101), dtype=np.int64)  # final column = failed/no prediction
    for row in rows:
        cm[row["true_label"], row["predicted_label"] if row["predicted_label"] is not None else 100] += 1
    tp = np.diag(cm[:, :100]).astype(float)
    support, predicted = cm.sum(1), cm[:, :100].sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(100), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(100), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(100), where=(precision + recall) != 0)
    total = int(support.sum())
    if total == 0:
        raise ValueError("Empty evaluation")
    metrics = {"top1_accuracy": float(tp.sum() / total), "macro_precision": float(precision.mean()),
               "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
               "weighted_f1": float((f1 * support).sum() / total), "total_videos": total,
               "failed_videos": int(cm[:, 100].sum()),
               "denominator": "all manifest test videos; errors count as incorrect/false negatives"}
    report = [{"label": i, "name": names[i], "precision": float(precision[i]), "recall": float(recall[i]),
               "f1": float(f1[i]), "support": int(support[i])} for i in range(100)]
    write_json(out / "metrics.json", metrics)
    write_csv(out / "classification_report.csv", report)
    write_json(out / "classification_report.json", report)
    norm = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float), where=support[:, None] != 0)
    np.save(out / "confusion_counts.npy", cm)
    np.save(out / "confusion_normalized.npy", norm)
    np.savetxt(out / "confusion_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out / "confusion_normalized.csv", norm, delimiter=",")
    write_json(out / "confusion_labels.json", {"rows": names, "columns": names + ["ERROR/ABSTAIN"]})
    pairs = sorted([{"true_label": names[i], "predicted_label": names[j], "count": int(cm[i, j])}
                    for i in range(100) for j in range(100) if i != j and cm[i, j]],
                   key=lambda x: -x["count"])
    write_csv(out / "top_confusions.csv", pairs[:30], ["true_label", "predicted_label", "count"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, data in (("counts", cm), ("normalized", norm)):
        fig, ax = plt.subplots(figsize=(24, 22))
        image = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(101), names + ["ERROR"], rotation=90, fontsize=6)
        ax.set_yticks(range(100), names, fontsize=6)
        ax.set(xlabel="Predicted", ylabel="True", title=f"Test confusion matrix ({name})")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.png", dpi=150)
        plt.close(fig)
    return metrics

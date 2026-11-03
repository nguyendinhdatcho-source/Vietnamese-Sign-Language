"""Build the method and results report from saved results without modifying application code."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
RUNS = ["m0_diff_avg"]
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def verify_sources():
    splits = pd.read_csv(ROOT / "vsl_motion_work/splits.csv")
    names = json.loads(
        (ROOT / "vsl_motion_work/splits_classes.json").read_text(encoding="utf-8")
    )
    labels = np.arange(len(names))
    assert len(names) == 100
    assert len(splits) == 3866 and not splits.path.duplicated().any()
    result = {"data": {}, "runs": {}}
    frames = {}
    for split, n in [("train", 3077), ("val", 385), ("test", 404)]:
        sub = splits[splits.split == split]
        counts = sub.groupby("label").size()
        assert len(sub) == n and len(counts) == 100
        result["data"][split] = {
            "videos": n,
            "classes": len(counts),
            "min_per_class": int(counts.min()),
            "max_per_class": int(counts.max()),
            "single_sample_classes": int((counts == 1).sum()),
        }
    for run in RUNS:
        folder = ROOT / "vsl_motion_work/runs" / run
        stored = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        history = pd.read_csv(folder / "history.csv")
        assert not history.epoch.duplicated().any()
        assert np.isclose(history.val_macro_f1.max(), stored["val"]["macro_f1"])
        entry = {
            "config": stored["config"],
            "epochs_completed": len(history),
            "best_epoch_one_based": int(history.loc[history.val_macro_f1.idxmax(), "epoch"]) + 1,
            "model_ms_per_clip": stored["model_ms_per_clip"],
            "device": stored["device"],
        }
        for split in ["val", "test"]:
            sub = splits[splits.split == split]
            predictions = pd.read_csv(folder / f"{split}_pred.csv")
            probs = np.load(folder / f"{split}_probs.npy")
            saved_cm = np.load(folder / f"{split}_confusion.npy")
            assert list(predictions.path) == list(sub.path), (run, split, "paths")
            assert probs.shape == (len(sub), 100)
            assert np.isfinite(probs).all()
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
            pred = probs.argmax(axis=1)
            assert np.array_equal(pred, predictions.pred)
            truth = sub.label.to_numpy()
            computed = {
                "top1": accuracy_score(truth, pred),
                "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
                "macro_precision": precision_score(truth, pred, labels=labels, average="macro", zero_division=0),
                "macro_recall": recall_score(truth, pred, labels=labels, average="macro", zero_division=0),
            }
            for key, value in computed.items():
                assert np.isclose(value, stored[split][key], atol=1e-12), (run, split, key)
            cm = confusion_matrix(truth, pred, labels=labels)
            assert np.array_equal(cm, saved_cm)
            support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
            per_f1 = 2 * tp / (support + predicted)
            entry[split] = dict(
                computed,
                correct=int(tp.sum()),
                errors=int(cm.sum() - tp.sum()),
                recall_one_classes=int(((tp == support) & (support > 0)).sum()),
                perfect_classes=int(((tp == support) & (tp == predicted) & (support > 0)).sum()),
                zero_f1_classes=[names[i] for i in labels if per_f1[i] == 0],
            )
        result["runs"][run] = entry
        frames[run] = (stored, history)
    result["source_sha256"] = {}
    paths = [
        ROOT / "make_splits.py",
        ROOT / "precompute_motion.py",
        ROOT / "train_motion.py",
        ROOT / "vsl_motion_work/splits.csv",
        ROOT / "vsl_motion_work/splits_classes.json",
    ]
    for run in RUNS:
        folder = ROOT / "vsl_motion_work/runs" / run
        paths.extend(folder / f for f in ["metrics.json", "history.csv", "test_probs.npy", "val_probs.npy"])
    for path in paths:
        result["source_sha256"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "verified_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return names, frames


def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def build_figures(names, frames):
    stored, history = frames[RUNS[0]]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, label, color="#EDF3F8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009", facecolor=color, edgecolor="#7890A3", linewidth=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#526779", lw=1.4))

    top = [
        (0.015, 0.16, "Video\n224 × 224"),
        (0.22, 0.22, "Loại frame\nđồng màu\nChuyển ảnh xám"),
        (0.49, 0.21, "Frame difference\nKhuếch đại × 4"),
        (0.75, 0.23, "Lấy mẫu trên ảnh diff\nLặp 3 kênh\nChuẩn hóa [−1, 1]"),
    ]
    for x, w, text in top:
        box(x, 0.75, w, 0.2, text)
    for (x, w, _), (nx, _, _) in zip(top[:-1], top[1:]):
        arrow((x + w + 0.01, 0.85), (nx - 0.012, 0.85))
    box(0.71, 0.23, 0.27, 0.24, "TimeDistributed\nMobileNetV2–ImageNet\n16 × 1.280")
    arrow((0.865, 0.74), (0.865, 0.48))
    box(0.365, 0.23, 0.29, 0.24, "Dropout 0,3\nTrung bình thời gian\nVector 1.280 chiều")
    arrow((0.70, 0.35), (0.665, 0.35))
    box(0.015, 0.23, 0.29, 0.24, "Dropout 0,4 → Dense 100\nSoftmax → Nhãn ký hiệu", "#EAF4EE")
    arrow((0.355, 0.35), (0.315, 0.35))
    ax.text(0.5, 0.045, "Chọn checkpoint theo Macro-F1 validation", fontsize=11, color="#22577A", ha="center")
    save_figure(fig, "01_pipeline")

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    x = np.arange(4)
    keys = ["top1", "macro_precision", "macro_recall", "macro_f1"]
    series = [("Validation", "val", "#7EA1BA"), ("Test", "test", "#22577A")]
    for i, (label, split, color) in enumerate(series):
        values = [stored[split][key] * 100 for key in keys]
        bars = ax.bar(x + (i - 0.5) * 0.35, values, width=0.32, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"], ylabel="Chỉ số (%)", ylim=(0, 110))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "02_metric_comparison")

    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.plot(history.epoch + 1, history.val_macro_f1 * 100, label="Macro-F1 validation", color="#22577A", marker="o", ms=3, lw=1.6)
    best = history.loc[history.val_macro_f1.idxmax()]
    bx, by = best.epoch + 1, best.val_macro_f1 * 100
    ax.scatter(bx, by, s=90, facecolor="white", edgecolor="#22577A", zorder=5, linewidth=1.6)
    ax.annotate(f"Epoch {int(bx)}: {by:.2f}%", xy=(bx, by), xytext=(bx - 7, by - 13), arrowprops=dict(arrowstyle="->", color="#526779"), fontsize=10)
    ax.axvline(3.5, color="#8B9299", ls="--", lw=1)
    ax.text(4.0, 8, "Bắt đầu fine-tune", color="#5E6266", fontsize=9)
    ax.set(xlabel="Epoch (tính từ 1)", ylabel="Macro-F1 validation (%)", xlim=(0, 34), ylim=(0, 90))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "03_validation_curves")

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    cm = np.load(ROOT / "vsl_motion_work/runs" / RUNS[0] / "test_confusion.npy").copy()
    np.fill_diagonal(cm, 0)
    pairs = np.argwhere(cm > 0)
    pairs = sorted(pairs, key=lambda p: (-cm[tuple(p)], names[p[0]], names[p[1]]))[:8]
    text = [f"{names[i]} → {names[j]}" for i, j in pairs]
    counts = [cm[i, j] for i, j in pairs]
    bars = ax.barh(text, counts, color="#22577A", height=0.62)
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.invert_yaxis()
    ax.set(xlim=(0, 2.6), xticks=range(3), xlabel="Số video bị nhầm trên test")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "04_top_confusions")


def set_font(style, name, size, color=None, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if color:
        style.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def add_table(doc, lines):
    parsed = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
    rows = [parsed[0]] + parsed[2:]
    table = doc.add_table(rows=0, cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cell = cells[j]
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(10 if len(cells) >= 6 else 10.5)
                    if i == 0:
                        r.bold = True
                        r.font.color.rgb = RGBColor.from_string("FFFFFF")
            if i == 0:
                set_cell_shading(cell, "22577A")
            elif i % 2:
                set_cell_shading(cell, "F0F4F7")
        row_properties = table.rows[-1]._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        row_properties.append(no_split)
        if i == 0:
            repeat = OxmlElement("w:tblHeader")
            row_properties.append(repeat)
    if len(rows[0]) == 2:
        for row in table.rows:
            row.cells[0].width = Cm(6.6)
            row.cells[1].width = Cm(10.2)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def build_docx(filename="bao_cao_phuong_phap_va_thuc_nghiem.docx"):
    source = OUT / "bao_cao_phuong_phap_va_thuc_nghiem.md"
    lines = source.read_text(encoding="utf-8").splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
    sec.left_margin, sec.right_margin = Cm(2.2), Cm(2.0)
    sec.header_distance = Cm(0.8)
    sec.footer_distance = Cm(0.8)
    set_font(doc.styles["Normal"], "Times New Roman", 12)
    normal = doc.styles["Normal"].paragraph_format
    normal.line_spacing = 1.15
    normal.space_after = Pt(7)
    normal.widow_control = True
    set_font(doc.styles["Title"], "Times New Roman", 17, "22577A", True)
    set_font(doc.styles["Heading 1"], "Times New Roman", 15, "22577A", True)
    set_font(doc.styles["Heading 2"], "Times New Roman", 13, "22577A", True)
    set_font(doc.styles["Caption"], "Times New Roman", 10, "52616B")
    for name in ["Heading 1", "Heading 2"]:
        doc.styles[name].paragraph_format.space_before = Pt(12)
        doc.styles[name].paragraph_format.space_after = Pt(7)
    header = sec.header.paragraphs[0]
    header.text = "VSL · FRAME DIFFERENCE + MOBILENETV2"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string("6B7785")
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Trang ").font.size = Pt(9)
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)
    doc.core_properties.title = "Nhận dạng ngôn ngữ ký hiệu Việt Nam dựa trên frame difference"
    doc.core_properties.subject = "Mô tả phương pháp và kết quả thực nghiệm"
    doc.core_properties.keywords = "VSL, frame difference, MobileNetV2, Macro-F1"
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            p = doc.add_paragraph(line[2:], "Title")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            p = doc.add_paragraph(line[3:], "Heading 1")
            if line.startswith("## 2."):
                p.paragraph_format.page_break_before = True
        elif line.startswith("### "):
            p = doc.add_paragraph(line[4:], "Heading 2")
        elif line.startswith("|"):
            chunk = []
            while i < len(lines) and lines[i].startswith("|"):
                chunk.append(lines[i])
                i += 1
            add_table(doc, chunk)
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            assert match, line
            caption, rel = match.groups()
            path = OUT / rel
            assert path.exists(), path
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(path), width=Cm(16.4))
            p = doc.add_paragraph(caption, "Caption")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("    "):
            p = doc.add_paragraph(line.strip())
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(5)
            for run in p.runs:
                run.font.name = "Cambria"
                run.font.size = Pt(11)
        else:
            p = doc.add_paragraph()
            for k, part in enumerate(re.split(r"\*\*", line)):
                if part:
                    p.add_run(part).bold = k % 2 == 1
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        i += 1
    destination = OUT / filename
    doc.save(destination)
    reopened = Document(destination)
    assert len(reopened.inline_shapes) == 4
    assert len(reopened.tables) == 7, len(reopened.tables)
    assert sum(p.style.name == "Heading 1" for p in reopened.paragraphs) == 2
    assert sum(p.style.name == "Heading 2" for p in reopened.paragraphs) == 13
    assert not any("TODO" in p.text or "TBD" in p.text for p in reopened.paragraphs)
    content = "\n".join(lines)
    assert not re.search(r"\bM[012]\b|baseline|optical flow|Conv1D|GRU|m0_diff|m1_flow|m2_diff", content, re.IGNORECASE)
    return destination


if __name__ == "__main__":
    class_names, run_frames = verify_sources()
    build_figures(class_names, run_frames)
    output = build_docx()
    print("Verified prediction probabilities, labels, confusion matrices and metrics on validation and test.")
    print(f"Created: {output}")
    print("Report contains 2 main sections, 13 subsections, 7 tables and 4 figures.")

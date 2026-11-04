"""ECE 491 - Python Lab 3: MNIST Digit Recognition Using a Perceptron.

Answers Q1-Q14 using the original four MNIST .gz archives.

Usage:
    python ECE491_Lab3_Submission.py
    python ECE491_Lab3_Submission.py --no-download --dataset-dir dataset
    python ECE491_Lab3_Submission.py --split-rule written-instruction

Outputs in lab3_results/:
    ECE491_Lab3_Experimental_Report.pdf  (measured answers, images and figures)
    ECE491_Lab3_Results.txt
    ECE491_Lab3_Results.json
    Q02_fifth_training.png
    Q03_fifth_test.png
    Q10_eta_comparison.png
    Q13_weight_visualisation.png

The professor's page-4 split CODE is the default. To use the last 20%
for validation (as described in prose), select --split-rule written-instruction.

The professor's early-stop condition is interpreted as accuracy FRACTION > 0.90.
No test examples are used for training, eta selection or early stopping.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")  # Save figures on desktop, Colab or headless runners.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# Original unmodified gzip checksums (published by the MNIST dataset maintainers).
OFFICIAL_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
NROWS = NCOLS = 28
SEED = 491


def require_archives(folder: Path, download: bool = True, check_md5: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = folder / filename
        if destination.is_file() and destination.stat().st_size > 0:
            if not check_md5 or md5sum(destination) == OFFICIAL_MD5[filename]:
                continue
            if not download:
                raise ValueError(
                    f"Not the original MNIST archive: {destination}. "
                    "Use the official four .gz files with matching checksums."
                )
            print(f"Existing archive has incorrect checksum; replacing: {filename}")
        if not download:
            raise FileNotFoundError(
                f"Required MNIST archive not found: {destination}\n"
                "Download the original four .gz files and put them in dataset/."
            )
        print(f"Downloading original MNIST file: {filename}")
        try:
            urlretrieve(MNIST_MIRROR + filename, destination)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {filename}. Upload the four original MNIST "
                "archives into dataset/, then run with --no-download."
            ) from exc
        if check_md5 and md5sum(destination) != OFFICIAL_MD5[filename]:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Downloaded archive failed official MD5 check: {filename}")


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Incomplete image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, cols) != (NROWS, NCOLS):
            raise ValueError(f"Invalid IDX image header in {path}")
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    if arr.size != count * rows * cols:
        raise ValueError(f"Unexpected image payload size in {path}")
    return arr.reshape(count, rows * cols).copy()


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Incomplete label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid IDX label header in {path}")
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    if arr.size != count:
        raise ValueError(f"Unexpected label payload size in {path}")
    return arr.copy()


def load_data(folder: Path):
    # Preserve professor's four variable names and source data layout.
    train_ims = read_images(folder / FILES[0])
    train_lbs = read_labels(folder / FILES[1])
    test_ims = read_images(folder / FILES[2])
    test_lbs = read_labels(folder / FILES[3])
    return train_ims, train_lbs, test_ims, test_lbs


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    # The laboratory defines sign(w^T*x). At the boundary (score == 0),
    # choose +1 to ensure the predictor has only the allowed class labels.
    return np.where(np.asarray(X) @ w >= 0.0, 1.0, -1.0)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.mean(predict(X, w) == y))


def train_once(X_train, y_train, X_val, y_val, eta, initial_w):
    """Perceptron exactly following professor's eta*error*x update.

    Validate after indices 0,100,...; stop at fractional accuracy > 0.90.
    """
    w = initial_w.copy()
    history = []
    stop_index = None
    for idx, (x, y_true) in enumerate(zip(X_train, y_train)):
        score = float(w @ x)
        y_pred = 1.0 if score >= 0.0 else -1.0
        error = float(y_true - y_pred)
        w += float(eta) * error * x
        if idx % 100 == 0:
            val_acc = accuracy(X_val, y_val, w)
            history.append((idx, val_acc))
            if val_acc > 0.90:
                stop_index = idx
                break
    final_val_acc = accuracy(X_val, y_val, w)
    return {
        "eta": float(eta),
        "weights": w,
        "history": history,
        "stop_index": stop_index,
        "steps_processed": idx + 1,
        "validation_accuracy": final_val_acc,
    }


def figure_digit(pixels, digit_label, caption, output):
    fig, ax = plt.subplots(figsize=(4.3, 4.3))
    ax.imshow(np.asarray(pixels).reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{caption}  |  Label = {int(digit_label)}", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def pdf_experimental_report(output, summary, eta_results, image_paths):
    """Generate actual-data experiment PDF from measured values only."""
    with PdfPages(output) as pdf:
        # Page 1: machine-generated measured numerical results.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.085, 0.945, "ECE 491  |  PYTHON LAB 3", fontsize=18, weight="bold")
        fig.text(0.085, 0.914, "Measured MNIST Perceptron Results", fontsize=12)
        fig.text(0.085, 0.878, "These measurements use the original four MNIST IDX archives.", fontsize=9.5)
        lines = [
            ("Q1", "Original shapes: train images (60000,784), labels (60000,); test images (10000,784), labels (10000,)"),
            ("Q2", f"Fifth training image label: {summary['fifth_training_label']}"),
            ("Q3", f"Fifth test image label: {summary['fifth_test_label']}"),
            ("Q4", f"Digits 0/1: {summary['filtered_train']} original training examples"),
            ("Q5", f"Digits 0/1: {summary['filtered_test']} original test examples"),
            ("Q6", f"Training = {summary['training_count']}; validation = {summary['validation_count']}"),
            ("Q7", "Images and labels cast to float32 in all splits"),
            ("Q8", "Labels vectorised: original 0 to -1; original 1 to +1"),
            ("Q9", "Perceptron implemented: 784 weights; update w <- w + eta*(y-yhat)*x"),
            ("Q10", f"Best tested eta: {summary['selected_eta']:.0e}"),
            ("Q11", f"Stop index: {summary['stop_index']} (0-based); samples processed: {summary['steps_processed']}"),
            ("Q12", f"Correct test images: {summary['test_correct']}/{summary['filtered_test']}; accuracy = {summary['test_accuracy_percent']:.4f}%"),
            ("Q13", "Learned weight map shown on page 3"),
            ("Q14", "Positive weights favor digit 1; negative weights favor digit 0"),
        ]
        y = 0.832
        for q, line in lines:
            fig.text(0.085, y, q, weight="bold", fontsize=10, va="top")
            # Wrap with textwrap but one/two lines only per item.
            from textwrap import wrap
            fragments = wrap(line, 91)
            for j, fragment in enumerate(fragments):
                fig.text(0.145, y - j * 0.019, fragment, fontsize=9.2, va="top")
            y -= 0.039 + 0.019 * (len(fragments) - 1)
        fig.text(0.085, 0.130, f"Split convention: {summary['split_rule']}", fontsize=9.3)
        fig.text(0.085, 0.105, "Learning rate is selected using validation data only; test data are held out.", fontsize=9.2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: actual fifth training and test images + learning rate plot.
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        from matplotlib.image import imread
        axes[0, 0].imshow(imread(image_paths[0]))
        axes[0, 0].set_title("Q2  |  Fifth training image")
        axes[0, 1].imshow(imread(image_paths[1]))
        axes[0, 1].set_title("Q3  |  Fifth test image")
        axes[1, 0].imshow(imread(image_paths[2]))
        axes[1, 0].set_title("Q10  |  Learning-rate comparison")
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.axis("off")
        axes[1, 1].axis("off")
        axes[1, 1].text(0.03, 0.94, "Validation stopping rule", fontsize=11, weight="bold", transform=axes[1,1].transAxes)
        axes[1, 1].text(0.03, 0.84, r"Stop at first checkpoint with accuracy > 90%.", fontsize=9.5, transform=axes[1,1].transAxes)
        axes[1, 1].text(0.03, 0.72, "Checks occur at indices 0, 100, 200, ...", fontsize=9.3, transform=axes[1,1].transAxes)
        fig.suptitle("ECE 491 Lab 3  |  Sample Images and Validation", fontsize=15, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: full-size learned weight map + plain-language interpretation.
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.1, 0.28, 0.8, 0.62])
        ax.imshow(imread(image_paths[3]))
        ax.axis("off")
        fig.text(0.085, 0.955, "Q13-14  |  Learned Perceptron Weights", fontsize=16, weight="bold")
        fig.text(0.085, 0.215, "Explanation", fontsize=11, weight="bold")
        fig.text(0.085, 0.185, "Positive pixel weights contribute towards the +1 (digit 1) class.", fontsize=9.5)
        fig.text(0.085, 0.162, "Negative pixel weights contribute towards the -1 (digit 0) class.", fontsize=9.5)
        fig.text(0.085, 0.139, "The image is a learned linear discriminant, not a literal digit.", fontsize=9.5)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--output-dir", type=Path, default=Path("lab3_results"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--allow-test-data", action="store_true",
                    help="For software smoke testing ONLY; do not use for student submissions.")
    ap.add_argument("--split-rule", choices=["professor-code", "written-instruction"], default="professor-code")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--etas", default="1e-7,1e-6,1e-5,1e-4,1e-3")
    args = ap.parse_args()

    require_archives(args.dataset_dir, download=not args.no_download,
                     check_md5=not args.allow_test_data)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # Q1: raw MNIST variables before any filtering.
    train_ims, train_lbs, test_ims, test_lbs = load_data(args.dataset_dir)
    shape_dict = {n: list(a.shape) for n, a in [
        ("train_ims", train_ims), ("train_lbs", train_lbs),
        ("test_ims", test_ims), ("test_lbs", test_lbs)
    ]}
    assert train_ims.shape == (60000, 784) and train_lbs.shape == (60000,)
    assert test_ims.shape == (10000, 784) and test_lbs.shape == (10000,)
    print("Q1: four original array shapes:", shape_dict)

    # Q2-Q3: fifth means Python index 4; before digit filtering.
    train_fifth = int(train_lbs[4])
    test_fifth = int(test_lbs[4])
    if not args.allow_test_data and (train_fifth, test_fifth) != (9, 4):
        raise ValueError("Original MNIST ordering is unexpected; check the input archives.")
    figure_digit(train_ims[4], train_fifth, "Q2 - Fifth training image", output / "Q02_fifth_training.png")
    figure_digit(test_ims[4], test_fifth, "Q3 - Fifth test image", output / "Q03_fifth_test.png")
    print(f"Q2: fifth original training image label = {train_fifth}")
    print(f"Q3: fifth original test image label = {test_fifth}")

    # Q4-Q5: filter to only 0/1 as in the handout.
    train_mask = np.logical_or(train_lbs == 0, train_lbs == 1)
    test_mask = np.logical_or(test_lbs == 0, test_lbs == 1)
    train_ims, train_lbs = train_ims[train_mask], train_lbs[train_mask]
    test_ims, test_lbs = test_ims[test_mask], test_lbs[test_mask]
    filtered_train, filtered_test = len(train_lbs), len(test_lbs)
    assert filtered_train == 12665 and filtered_test == 2115
    print(f"Q4: filtered training count = {filtered_train}")
    print(f"Q5: filtered test count = {filtered_test}")

    # Q6: preserve professor's literal code by default. Explain conflict in PDF.
    split = int(0.8 * filtered_train)
    if args.split_rule == "professor-code":
        val_ims, val_lbs = train_ims[:split], train_lbs[:split]
        train_ims, train_lbs = train_ims[split:], train_lbs[split:]
    else:
        val_ims, val_lbs = train_ims[split:], train_lbs[split:]
        train_ims, train_lbs = train_ims[:split], train_lbs[:split]

    # Q7: six arrays to float32; do not rescale 0..255 (not requested).
    train_ims, train_lbs = train_ims.astype(np.float32), train_lbs.astype(np.float32)
    val_ims, val_lbs = val_ims.astype(np.float32), val_lbs.astype(np.float32)
    test_ims, test_lbs = test_ims.astype(np.float32), test_lbs.astype(np.float32)
    assert all(a.dtype == np.float32 for a in (train_ims, train_lbs, val_ims, val_lbs, test_ims, test_lbs))
    print("Q6: training / validation / test:", len(train_lbs), len(val_lbs), len(test_lbs))
    print("Q7: all six arrays cast to float32.")

    # Q8: vectorised mapping without any for loop.
    train_lbs = 2.0 * train_lbs - 1.0
    val_lbs = 2.0 * val_lbs - 1.0
    test_lbs = 2.0 * test_lbs - 1.0
    assert all(np.all(np.isin(y, [-1.0, 1.0])) for y in (train_lbs, val_lbs, test_lbs))
    print("Q8: label mapping 0 -> -1, 1 -> +1 verified.")

    # Q9-Q10: experiment with common initialization; pick best validation acc.
    etas = [float(s) for s in args.etas.split(",") if s.strip()]
    if not etas or any(not np.isfinite(v) or v <= 0 for v in etas):
        raise ValueError("--etas must contain finite positive numbers.")
    initial_w = np.random.default_rng(args.seed).normal(0, 1, size=784)
    experiments = []
    for eta in etas:
        result = train_once(train_ims, train_lbs, val_ims, val_lbs, eta, initial_w)
        experiments.append(result)
        print(f"Q10: eta={eta:.0e}, validation={100*result['validation_accuracy']:.3f}%, stop={result['stop_index']}")
    best = max(experiments, key=lambda r: (r["validation_accuracy"], -r["eta"]))
    learned_w = best["weights"]

    # Q11/Q12: threshold is strictly >0.90; test used only after eta chosen.
    test_pred = predict(test_ims, learned_w)
    test_correct = int(np.count_nonzero(test_pred == test_lbs))
    test_accuracy_percent = 100.0 * test_correct / len(test_lbs)
    print(f"Q11: stop index = {best['stop_index']}; samples processed = {best['steps_processed']}")
    print(f"Q12: test accuracy = {test_correct}/{len(test_lbs)} = {test_accuracy_percent:.4f}%")

    # Q10 chart of validation accuracy at the final selected checkpoint.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar([f"{e['eta']:.0e}" for e in experiments], [100*e["validation_accuracy"] for e in experiments])
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% stopping threshold")
    ax.set_xlabel("Tested learning rate eta")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Q10 - Learning-rate comparison")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "Q10_eta_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Q13: visualize signed 28x28 weights on symmetric scale.
    weight_img = learned_w.reshape(28, 28)
    maxabs = max(1e-12, float(np.max(np.abs(weight_img))))
    fig, ax = plt.subplots(figsize=(6.1, 5.4))
    im = ax.imshow(weight_img, cmap="bwr", vmin=-maxabs, vmax=maxabs)
    ax.set_title("Q13 - Learned perceptron weights")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Signed weight: red +1, blue -1")
    fig.tight_layout()
    fig.savefig(output / "Q13_weight_visualisation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "original_shapes": shape_dict,
        "fifth_training_label": train_fifth,
        "fifth_test_label": test_fifth,
        "filtered_train": filtered_train,
        "filtered_test": filtered_test,
        "split_rule": args.split_rule,
        "dataset_authenticity": "OFFICIAL MNIST MD5 VERIFIED" if not args.allow_test_data else "NONSTANDARD TEST DATA - DO NOT SUBMIT",
        "training_count": int(len(train_lbs)),
        "validation_count": int(len(val_lbs)),
        "selected_eta": best["eta"],
        "stop_index": best["stop_index"],
        "steps_processed": best["steps_processed"],
        "validation_accuracy_percent": 100.0*best["validation_accuracy"],
        "test_correct": test_correct,
        "test_accuracy_percent": test_accuracy_percent,
        "eta_experiments": [
            {k: e[k] for k in ("eta", "stop_index", "steps_processed", "validation_accuracy")}
            for e in experiments
        ],
    }
    (output / "ECE491_Lab3_Results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "ECE 491 - Python Lab 3 | ANSWERS FROM ORIGINAL MNIST DATA",
        "=" * 60,
    ]
    for key, val in summary.items():
        lines.append(f"{key}: {val}")
    (output / "ECE491_Lab3_Results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pdf_experimental_report(
        output / "ECE491_Lab3_Experimental_Report.pdf",
        summary,
        experiments,
        [output / x for x in (
            "Q02_fifth_training.png", "Q03_fifth_test.png",
            "Q10_eta_comparison.png", "Q13_weight_visualisation.png"
        )],
    )
    print("\nGenerated measured results in:", output.resolve())
    if args.allow_test_data:
        print("WARNING: These results use synthetic test data and must NOT be submitted as MNIST results.")
    return summary


if __name__ == "__main__":
    main()

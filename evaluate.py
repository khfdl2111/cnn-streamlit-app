# evaluate.py
# Evaluate on an external test set and save a report + confusion matrix plot.
# Usage:
#   python evaluate.py --test_dir data/test --artifacts_dir artifacts
#
# test_dir structure:
# data/test/
#   class_1/*.jpg
#   class_2/*.jpg
#
import argparse, os, json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    return p.parse_args()

def load_artifacts(artifacts_dir):
    model = tf.keras.models.load_model(Path(artifacts_dir)/"model.keras")
    classes = json.loads(Path(artifacts_dir, "class_names.json").read_text())
    return model, classes

def iter_images(root, class_names):
    for ci, cname in enumerate(class_names):
        folder = Path(root)/cname
        if not folder.exists():
            continue
        for p in folder.glob("**/*"):
            if p.suffix.lower() in [".jpg",".jpeg",".png"]:
                yield ci, cname, p

def preprocess(img, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return np.expand_dims(x, axis=0)

def main():
    args = parse_args()
    model, classes = load_artifacts(args.artifacts_dir)
    H, W = int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2])

    y_true, y_pred = [], []
    paths = []

    for true_idx, cname, path in iter_images(args.test_dir, classes):
        img = Image.open(path)
        x = preprocess(img, (W, H))
        prob = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(prob))
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        paths.append(str(path))

    if not y_true:
        print("No test images found.")
        return

    # Metrics
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save report
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval_report.json").write_text(json.dumps(report, indent=2))

    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir/"confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved eval report to:", out_dir/"eval_report.json")
    print("Saved confusion matrix to:", out_dir/"confusion_matrix.png")

if __name__ == "__main__":
    main()

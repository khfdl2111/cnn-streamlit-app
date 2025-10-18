# train_transfer.py
# Transfer learning + augmentation + BatchNorm + Dropout
# Optional hyperparameter tuning with keras-tuner (Hyperband)
#
# Usage:
#   python train_transfer.py --data_dir data --epochs 10
#   python train_transfer.py --data_dir data --tune --max_epochs 15
#
# Expect directory:
# data/
#   train/class_1, class_2, ...
#   val/class_1, class_2, ...

import argparse, json, os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Optional tuner
try:
    import keras_tuner as kt
except Exception:
    kt = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--max_epochs", type=int, default=15)
    p.add_argument("--output_dir", type=str, default="artifacts")
    return p.parse_args()

def load_ds(root, img, bs):
    train_dir = Path(root)/"train"
    val_dir   = Path(root)/"val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expect data/train and data/val with class subfolders.")
    train = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, seed=42, image_size=(img, img), batch_size=bs)
    val   = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, seed=42, image_size=(img, img), batch_size=bs)
    AUTOTUNE = tf.data.AUTOTUNE
    return (train.cache().shuffle(1000).prefetch(AUTOTUNE),
            val.cache().prefetch(AUTOTUNE),
            train.class_names)

def build_model(img, ncls, hp=None):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img, img, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(img, img, 3))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Defaults
    dropout = 0.3
    units = 256
    lr = 1e-3
    if hp is not None:
        dropout = hp.Float("dropout", 0.2, 0.6, step=0.1, default=0.3)
        units = hp.Choice("units", [128, 256, 384, 512], default=256)
        lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4], default=1e-3)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(ncls, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def tune_model(train, val, img, ncls, max_epochs):
    if kt is None:
        raise RuntimeError("Install keras-tuner to use --tune (pip install keras-tuner).")
    def builder(hp):
        return build_model(img, ncls, hp)
    tuner = kt.Hyperband(builder, objective="val_accuracy",
                         max_epochs=max_epochs, factor=3,
                         directory="kt_dir", project_name="mobilenetv2",
                         overwrite=True)
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                       restore_best_weights=True)
    tuner.search(train, validation_data=val, callbacks=[es])
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hp)
    return model, best_hp

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train, val, classes = load_ds(args.data_dir, args.img_size, args.batch_size)
    ncls = len(classes)

    if args.tune:
        model, best_hp = tune_model(train, val, args.img_size, ncls, args.max_epochs)
        print("Best HP:", {k: best_hp.get(k) for k in best_hp.values.keys()})
    else:
        model = build_model(args.img_size, ncls)

    cbs = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "model.keras"),
            save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    hist = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=cbs)

    # Optional light fine-tuning: unfreeze top layers
    base = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if base:
        base_model = base[0]
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=3, callbacks=cbs)

    model.save(os.path.join(args.output_dir, "model_last.keras"))
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(classes, f, indent=2)
    loss, acc = model.evaluate(val, verbose=0)
    print(f"Val Acc: {acc:.4f} | Val Loss: {loss:.4f}")
    print("Saved to:", args.output_dir)

if __name__ == "__main__":
    main()

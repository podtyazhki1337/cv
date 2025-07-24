# train.py
import os
import argparse
import tensorflow as tf
ldpaths = [
    os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib"),
    "/usr/lib/x86_64-linux-gnu"
]
os.environ["LD_LIBRARY_PATH"] = ":".join(ldpaths)
print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow import keras as K

import settings
from dataset_tif import TIFPatchDataset
from model_classic import unet_3d, dice_loss, dice_coef, soft_dice_coef

# ────────────── argparse ────────────────────────────────────────────────
parser = argparse.ArgumentParser("Train 3‑D U‑Net")
parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
parser.add_argument("--filters", type=int, default=settings.FILTERS)
parser.add_argument("--use_upsampling", action="store_true",
                    default=settings.USE_UPSAMPLING)
parser.add_argument("--validation_split", type=float,
                    default=settings.VALIDATION_SPLIT)
args = parser.parse_args()

# ---------- размеры тайла -------------------------------------------------
tile_h = min(256, settings.TIF_HEIGHT)
tile_w = min(256, settings.TIF_WIDTH)
tile_d = settings.TILE_DEPTH                      # уже кратно 16

crop_dim = (tile_h, tile_w, tile_d)
input_shape = crop_dim + (settings.NUMBER_INPUT_CHANNELS,)

# ────────────── датасет ────────────────────────────────────────────────
tif_data = TIFPatchDataset(crop_dim=crop_dim,
                           batch_size=args.batch_size,
                           val_ratio=args.validation_split,
                           seed=settings.RANDOM_SEED,
                           img_dir=settings.TIF_IMG_DIR,
                           mask_dir=settings.TIF_MASK_DIR)

model = unet_3d(input_dim=input_shape,
                    filters=args.filters,
                    number_output_classes=settings.NUMBER_OUTPUT_CLASSES,
                    use_upsampling=args.use_upsampling)

model.compile(
    optimizer=K.optimizers.Adam(),
    loss=dice_loss,                 # остаётся как был
    metrics=[dice_coef]             # одной метрики достаточно
)

# ────────────── колбэки ────────────────────────────────────────────────
ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
best_ckpt = os.path.join(ckpt_dir, "unet_hela_best")
last_ckpt = os.path.join(ckpt_dir, "unet_hela_last")

checkpoint_best = K.callbacks.ModelCheckpoint(
        best_ckpt,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        save_weights_only=True,     # ←‑‑‑ главное изменение
        verbose=1
)
checkpoint_last = K.callbacks.ModelCheckpoint(
        last_ckpt,
        save_weights_only=True,     # ←‑‑‑ то же
        save_freq="epoch",
        verbose=1
)
earlystop = K.callbacks.EarlyStopping(
    monitor="val_dice_coef", mode="max", patience=40, restore_best_weights=True)

steps_per_epoch = (
    len(tif_data.train_ids)
    * (settings.TIF_HEIGHT // tile_h)
    * (settings.TIF_WIDTH // tile_w)
    // args.batch_size
)

# ────────────── обучение ───────────────────────────────────────────────
model.fit(tif_data.ds_train,
          validation_data=tif_data.ds_val,
          steps_per_epoch=steps_per_epoch,
          epochs=args.epochs,
          callbacks=[checkpoint_best, checkpoint_last, earlystop])

# ────────────── оценка ─────────────────────────────────────────────────
best_model = unet_3d(
        input_dim=input_shape,
        filters=args.filters,
        number_output_classes=settings.NUMBER_OUTPUT_CLASSES,
        use_upsampling=args.use_upsampling)
best_model.load_weights(best_ckpt)

print("\n\nEvaluating best model on the validation dataset.")
print("===============================================")
loss, dice_c, soft_dice_c = best_model.evaluate(tif_data.ds_val)
print(f"Average Dice Coefficient on validation dataset = {dice_c:.4f}")

# ────────────── финальный экспорт ──────────────────────────────────────
best_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
final_model_name = best_ckpt + "_final"
K.models.save_model(best_model, final_model_name, include_optimizer=False)

print("\nConvert to OpenVINO:")
print("source /opt/intel/openvino/bin/setupvars.sh")
print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
print(f"       --saved_model_dir {final_model_name} \\")
print(f"       --model_name {settings.SAVED_MODEL_NAME} \\")
print("       --batch 1  \\")
print("       --output_dir openvino_models/FP32 \\")
print("       --data_type FP32\n")

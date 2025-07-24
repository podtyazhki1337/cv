import sys, os, time, random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tensorflow import keras as K
import argparse           # всё то же
from dataset_tif import TIFPatchDataset  # <-- новый импорт
from model_DR import unet_3d, dice_loss, dice_coef, soft_dice_coef
import settings
import os
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
                                    # 256,256,64,1

# 1) датасет ждёт 3-элементный crop_dim
tif_data = TIFPatchDataset(crop_dim=crop_dim,
                           batch_size=args.batch_size,
                           val_ratio=args.validation_split,
                           seed=settings.RANDOM_SEED,
                           img_dir=settings.TIF_IMG_DIR,
                           mask_dir=settings.TIF_MASK_DIR)

# 2) модель ждёт 4-элементный input_shape
model = unet_3d(input_dim=input_shape,
                    filters=args.filters,
                    number_output_classes=settings.NUMBER_OUTPUT_CLASSES,
                    use_upsampling=args.use_upsampling)

model.compile(
    optimizer=K.optimizers.Adam(),
    loss=dice_loss,                 # остаётся как был
    metrics=[dice_coef]             # одной метрики достаточно
)


ckpt_dir = "checkpoints/unet_drres_hela"; os.makedirs(ckpt_dir, exist_ok=True)
best_ckpt = os.path.join(ckpt_dir, "unet_DRres_hela_best")
last_ckpt = os.path.join(ckpt_dir, "unet_DRres_hela_last")


# имена файлов

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
"""
4. Load best model on validation dataset and run on the test
dataset to show generalizability
"""


print("\n\nEvaluating best model on the testing dataset.")
print("=============================================")
loss, dice_c, soft_dice_c = best_model.evaluate(tif_data.ds_test)

print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))

"""
5. Save the best model without the custom objects (dice, etc.)
   NOTE: You should be able to do .load_model(compile=False), but this
   appears to currently be broken in TF2. To compensate, we're
   just going to re-compile the model without the custom objects and
   save as a new model (with suffix "_final")
"""

best_model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                   optimizer="adam")
K.models.save_model(best_model, best_ckpt,
                    include_optimizer=False)

"""
Converting the model to OpenVINO
"""
print("\n\nConvert the TensorFlow model to OpenVINO by running:\n")
print("source /opt/intel/openvino/bin/setupvars.sh")
print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
print("       --saved_model_dir {} \\".format(best_ckpt))
print("       --model_name {} \\".format(args.saved_model_name))
print("       --batch 1  \\")
print("       --output_dir {} \\".format(os.path.join("openvino_models", "FP32")))
print("       --data_type FP32\n\n")
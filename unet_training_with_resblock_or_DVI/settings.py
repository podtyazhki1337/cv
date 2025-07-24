# settings.py
from pathlib import Path
from tifffile import imread
import math, os

# корневая папка датасета
DATA_PATH   = "/NAS/mmaiurov/Datasets/Hela_MRC"
TIF_IMG_DIR = Path(DATA_PATH) / "images"
TIF_MASK_DIR = Path(DATA_PATH) / "masks"
SAVED_MODEL_NAME = "unet_hela"

# --- определяем размеры первого TIFF ------------------------------------
def _get_tif_shape(dir_):
    for f in sorted(os.listdir(dir_)):
        if f.lower().endswith((".tif", ".tiff", ".ome.tif")):
            return imread(os.path.join(dir_, f)).shape
    raise FileNotFoundError("*.tif not found in " + dir_)

_shape = _get_tif_shape(str(TIF_IMG_DIR))          # (Z,Y,X) или (Z,Y,X,C)
if len(_shape) == 3:  TIF_DEPTH, TIF_HEIGHT, TIF_WIDTH = _shape
else:                 TIF_DEPTH, TIF_HEIGHT, TIF_WIDTH, _ = _shape

# --- тайл ----------------------------------------------------------------
TILE_HEIGHT = 256
TILE_WIDTH  = 256
TILE_DEPTH  = int(math.ceil(TIF_DEPTH / 16) * 16)   # 61→64, 40→48 …

# --- обучение ------------------------------------------------------------
EPOCHS               = 300
BATCH_SIZE           = 2
NUMBER_INPUT_CHANNELS  = 1
NUMBER_OUTPUT_CLASSES  = 1
VALIDATION_SPLIT     = 0.2
FILTERS              = 16
USE_UPSAMPLING       = False
PRINT_MODEL          = False
RANDOM_SEED          = 816
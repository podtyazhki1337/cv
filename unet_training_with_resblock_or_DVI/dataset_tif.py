# dataset_tif.py
import os, glob, random
import numpy as np
import tensorflow as tf
from tifffile import imread
import tensorflow_addons as tfa
import settings

# ---------- утилиты ------------------------------------------------------
def _z_norm(v):
    v = v.astype(np.float32)
    m, s = v.mean(), v.std()
    return (v - m) / (s if s > 1e-6 else 1.0)

def _pad_depth(vol, target):
    """
    vol  : (H,W,D)  или (H,W,D,1)
    target : желаемое D (кратно 16, например 48 или 64)

    • если D < target  →  зеркалкой допадим снизу
    • если D > target  →  центр‑кропнем по Z
    """
    d = vol.shape[2]
    if d < target:                           # допадить
        pad = target - d
        pad_cfg = ((0,0), (0,0), (0,pad)) if vol.ndim==3 \
                  else ((0,0), (0,0), (0,pad), (0,0))
        vol = np.pad(vol, pad_cfg, mode="reflect")

    elif d > target:                         # кропнуть центр
        start = (d - target) // 2
        end   = start + target
        if vol.ndim == 3:
            vol = vol[:, :, start:end]
        else:                                # 4‑D (H,W,D,C)
            vol = vol[:, :, start:end, :]

    return vol

def _gauss2d_slice(slice3d, sigma=0.5):
    """
    slice3d : (H, W, 1) – один XY‑срез
    Возвращает такой же срез после Gaussian‑blur.
    """
    # tfa.gaussian_filter2d ждёт 4‑D => добавляем batch=1
    slice4d = slice3d[None, ...]                  # (1, H, W, 1)
    blurred = tfa.image.gaussian_filter2d(
        slice4d,
        filter_shape=(3, 3),
        sigma=(sigma, sigma),
        padding='REFLECT'
    )
    return blurred[0]                             # обратно (H, W, 1)

def _augment(img, msk, p_blur=0.5):
    """
    img, msk : (H, W, D, 1)
    C вероятностью p_blur блюрит КАЖДЫЙ Z‑срез,
    сохраняя формы (H, W, D, 1).
    """
    def _blur_volume(vol):
        # берём все D срезов за раз: (H,W,D,1) -> (D, H, W, 1)
        vol_t = tf.transpose(vol, (2, 0, 1, 3))
        vol_t = tf.map_fn(
            _gauss2d_slice,
            vol_t,
            fn_output_signature=vol.dtype
        )
        # обратно (H,W,D,1)
        return tf.transpose(vol_t, (1, 2, 0, 3))

    do_blur = tf.random.uniform(()) < p_blur
    img = tf.cond(do_blur, lambda: _blur_volume(img), lambda: img)
    return img, msk

# ---------- датасет ------------------------------------------------------
class TIFPatchDataset:
    def __init__(self,
                 img_dir=settings.TIF_IMG_DIR,
                 mask_dir=settings.TIF_MASK_DIR,
                 crop_dim=(settings.TILE_HEIGHT,
                           settings.TILE_WIDTH,
                           settings.TILE_DEPTH),
                 batch_size=settings.BATCH_SIZE,
                 val_ratio=settings.VALIDATION_SPLIT,
                 seed=settings.RANDOM_SEED):

        self.H, self.W, self.D = crop_dim
        self.bs  = batch_size
        self.rnd = np.random.default_rng(seed)

        imgs  = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.tif*")))
        assert len(imgs) == len(masks), "Imgs!=Masks"

        ids = list(range(len(imgs)))
        random.seed(seed); random.shuffle(ids)
        n_val = int(len(ids) * val_ratio)
        self.val_ids, self.train_ids = ids[:n_val], ids[n_val:]
        self.imgs, self.masks = imgs, masks

        self.ds_train = self._make(self.train_ids, True)
        self.ds_val   = self._make(self.val_ids,   False, shuffle=False, repeat=False)

    # -- helpers ----------------------------------------------------------
    def _load(self, idx):
        img = imread(self.imgs[idx]);  msk = imread(self.masks[idx])
        img = np.moveaxis(img, 0, -1); msk = np.moveaxis(msk, 0, -1)
        img = _pad_depth(img, settings.TILE_DEPTH)
        msk = _pad_depth(msk, settings.TILE_DEPTH)
        return _z_norm(img)[...,None], (msk>0).astype(np.float32)[...,None]

    def _crop(self, img, msk):
        H,W,D,_ = img.shape
        x0 = self.rnd.integers(0, H-self.H+1)
        y0 = self.rnd.integers(0, W-self.W+1)
        return (img[x0:x0+self.H, y0:y0+self.W, :, :],
                msk[x0:x0+self.H, y0:y0+self.W, :, :])

    def _gen(self, id_list, augm, repeat):
        while True:
            idx = self.rnd.choice(id_list)
            img, msk = self._crop(*self._load(idx))
            img, msk = tf.convert_to_tensor(img), tf.convert_to_tensor(msk)
            if augm: img, msk = _augment(img, msk)
            yield img, msk
            if not repeat: break

    def _make(self, id_list, augm, shuffle=True, repeat=True):
        ds = tf.data.Dataset.from_generator(
            lambda: self._gen(id_list, augm, repeat),
            output_signature=(
                tf.TensorSpec((self.H,self.W,self.D,1), tf.float32),
                tf.TensorSpec((self.H,self.W,self.D,1), tf.float32))
        )
        if shuffle: ds = ds.shuffle(4*self.bs, seed=settings.RANDOM_SEED)
        if repeat:  ds = ds.repeat()
        return ds.batch(self.bs).prefetch(tf.data.AUTOTUNE)
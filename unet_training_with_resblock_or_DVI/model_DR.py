import sys, os, time, random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparser import args
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_addons as tfa
import numpy as np

tf.config.experimental_run_functions_eagerly(True)
# -----------------------------------------------------------------------------
# 1.  Метрики и базовая Dice-потеря (остались без изменений)
# -----------------------------------------------------------------------------
def dice_coef(target, prediction, axis=(1, 2, 3, 4),  # добавили канал
              smooth=1e-6, threshold=None):
    """
    Dice = 2|X∩Y| / (|X|+|Y|).  При threshold=None — soft‑вариант,
    при threshold=0.5 – «жёсткая» версия.
    """
    if threshold is not None:            # ←  хотим «жёсткий» вариант – задаём явно
        prediction = tf.cast(prediction > threshold, prediction.dtype)

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union        = tf.reduce_sum(target + prediction,  axis=axis)
    dice         = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def soft_dice_coef(target, prediction, axis=(1, 2, 3), smooth=1e-4):
    inter = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    return tf.reduce_mean((2. * inter + smooth) / (union + smooth))


def dice_loss(target, prediction, axis=(1, 2, 3), smooth=1e-4):
    inter = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    num = tf.reduce_mean(inter + smooth)
    den = tf.reduce_mean(t + p + smooth)
    return -tf.math.log(2.*num) + tf.math.log(den)


# -----------------------------------------------------------------------------
# 2.  Вспом-функция: поворачиваем каждые XY-срезы на 45 °
#     tfa.image.rotate сохраняет форму, «пустоту» заполняем отражением.
# -----------------------------------------------------------------------------
_ANGLE = tf.constant(np.pi / 4, tf.float32)   # 45°

def _rotate_vol(vol, angle=_ANGLE, pad_val=1.0):
    """
    vol : (B, D, H, W, C)  – именно такой порядок у Keras Conv3D
    (batch, depth, height, width, channels)

    Поворачиваем каждый XY‑срез, сохраняя исходную форму.
    """
    # 1. (B,D,H,W,C)  →  (B*D, H, W, C)
    shape = tf.shape(vol)
    b, d = shape[0], shape[1]
    flat = tf.reshape(vol, (b * d, shape[2], shape[3], shape[4]))

    rot = tfa.image.rotate(
        flat,
        angles=angle,
        interpolation='bilinear',
        fill_mode='reflect'
    )

    # 2. Обратно  (B, D, H, W, C)
    rot = tf.reshape(rot, shape)
    return rot

def _inv_rotate_vol(vol):
    return _rotate_vol(vol, angle=-_ANGLE, pad_val=0.0)


# ---------------------------------------------------------------------
# 3.  Динамический Crop + Concat слой (ПОЛНОСТЬЮ ПЕРЕПИСАН)
# ---------------------------------------------------------------------
# class CropAndConcat(K.layers.Layer):
#     """
#     Centrally crops (or pads) `up` to match spatial dims of `skip`
#     and concatenates them. Правильно обрабатывает размерности каналов.
#     """
#     def __init__(self, axis=-1, **kwargs):
#         super().__init__(**kwargs)
#         self.axis = axis
#         self.supports_masking = True
#
#     def build(self, input_shape):
#         up_shape, skip_shape = input_shape
#
#         # Проверяем, что размерности каналов определены
#         if up_shape[-1] is None or skip_shape[-1] is None:
#             raise ValueError(f"Channel dimensions must be defined. Got up_shape={up_shape}, skip_shape={skip_shape}")
#
#         self.up_channels = up_shape[-1]
#         self.skip_channels = skip_shape[-1]
#         self.output_channels = self.up_channels + self.skip_channels
#
#         super().build(input_shape)
#
#     def call(self, inputs):
#         up, skip = inputs
#
#         # Получаем динамические размеры
#         up_shape = tf.shape(up)
#         skip_shape = tf.shape(skip)
#
#         # Пространственные размерности [D, H, W]
#         up_spatial = up_shape[1:4]
#         skip_spatial = skip_shape[1:4]
#         diff = up_spatial - skip_spatial
#
#         # -------- 1. CROP `up` if larger ----------
#         crop_begin = tf.maximum(diff // 2, 0)
#         crop_size = tf.minimum(up_spatial, skip_spatial)
#
#         # Создаем индексы для slice
#         batch_size = up_shape[0]
#         channels = up_shape[4]
#
#         begin = tf.stack([0,
#                           crop_begin[0],
#                            crop_begin[1],
#                           crop_begin[2],
#                           0]  # 0‑й и 4‑й – batch, channels
#             )
#
#         size = tf.stack(
#                         [batch_size,
#                                        crop_size[0],
#                                        crop_size[1],
#                                        crop_size[2],
#                                        channels]
#             )
#
#         up_cropped = tf.slice(up, begin, size)
#
#         # -------- 2. PAD `up` if smaller ----------
#         pad_needed = tf.maximum(skip_spatial - up_spatial, 0)
#         pad_begin = pad_needed // 2
#         pad_end = pad_needed - pad_begin
#
#         paddings = tf.stack([
#             [0, 0],  # batch
#             [pad_begin[0], pad_end[0]],  # depth
#             [pad_begin[1], pad_end[1]],  # height
#             [pad_begin[2], pad_end[2]],  # width
#             [0, 0]   # channels
#         ])
#
#         up_padded = tf.pad(up_cropped, paddings, mode='REFLECT')
#
#         # -------- 3. CONCAT ------------------------
#         result = tf.concat([up_padded, skip], axis=self.axis)
#
#         return result
#
#     def compute_output_shape(self, input_shapes):
#         up_shape, skip_shape = input_shapes
#
#         # Берем пространственные размерности от skip
#         output_shape = list(skip_shape)
#
#         # Суммируем каналы
#         up_channels = up_shape[-1] if up_shape[-1] is not None else 0
#         skip_channels = skip_shape[-1] if skip_shape[-1] is not None else 0
#         output_shape[-1] = up_channels + skip_channels
#
#         return tuple(output_shape)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({'axis': self.axis})
#         return config
class CropAndConcat(K.layers.Layer):
    """
    Centrally crops (или pads) тензор `up` под spatial‑размеры `skip`
    и конкатенирует их по каналам.  Работает как для форм
    (B, D, H, W, C), так и для (B, H, W, D, C) — мы всегда берём
    оси 1:4 как «пространственные».

    ✔︎ никаких выходов за границы при tf.slice
    ✔︎ если `up` уже нужного размера — обрезка/паддинг пропускаются
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    # ────────────────────────────── build ──────────────────────────────
    def build(self, input_shape):
        up_shape, skip_shape = input_shape

        # Проверяем, что каналы известны
        if up_shape[-1] is None or skip_shape[-1] is None:
            raise ValueError(
                f"Channel dimensions must be defined. "
                f"Got up_shape={up_shape}, skip_shape={skip_shape}"
            )
        self.up_channels   = up_shape[-1]
        self.skip_channels = skip_shape[-1]
        self.output_channels = self.up_channels + self.skip_channels
        super().build(input_shape)

    # ────────────────────────────── call ───────────────────────────────
    def call(self, inputs):
        up, skip = inputs

        u_shape = tf.shape(up)      # (B, s1, s2, s3, C)
        s_shape = tf.shape(skip)

        # 1. central crop, если up крупнее
        crop = tf.maximum(u_shape[1:4] - s_shape[1:4], 0)
        crop_beg = crop // 2
        crop_size = tf.minimum(u_shape[1:4], s_shape[1:4])

        begin = tf.concat([[0], crop_beg, [0]], 0)          # len = 5
        size  = tf.concat([[-1], crop_size, [-1]], 0)       # -1 → вся ось
        up_cropped = tf.slice(up, begin, size)              # (B,*,*,*,C)

        # 2. symmetric pad, если up меньше
        pad = tf.maximum(s_shape[1:4] - tf.shape(up_cropped)[1:4], 0)
        pad_beg = pad // 2
        pad_end = pad - pad_beg
        paddings = tf.concat(
            [
                [[0, 0]],                                   # batch
                tf.stack([pad_beg, pad_end], axis=1),       # 3 spatial
                [[0, 0]]                                    # channels
            ],
            axis=0
        )
        up_aligned = tf.pad(up_cropped, paddings, mode='REFLECT')

        # 3. concat по канальному измерению
        result = tf.concat([up_aligned, skip], axis=self.axis)

        # ← ДОБАВИТЬ:
        static_shape = skip.shape[:-1] + (self.output_channels,)
        result.set_shape(static_shape)

        return result

    # ─────────────────────── auxiliary methods ─────────────────────────
    def compute_output_shape(self, input_shapes):
        up_shape, skip_shape = input_shapes
        # spatial берём от skip, каналы суммируем
        return (up_shape[0],
                *skip_shape[1:4],
                up_shape[-1] + skip_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'axis': self.axis})
        return cfg

# ---------------------------------------------------------------------
# 4.  Стандартный 2-conv блок
# ---------------------------------------------------------------------
def _conv_block(x, name, filters, params):
    x = K.layers.Conv3D(filters, **params, name=f"{name}_c0")(x)
    x = K.layers.BatchNormalization(name=f"{name}_bn0")(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv3D(filters, **params, name=f"{name}_c1")(x)
    x = K.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = K.layers.Activation('relu', name=name)(x)
    return x


# ---------------------------------------------------------------------
# 5.  Ядро U-Net-3D (возвращает inp → pred) (ИСПРАВЛЕН)
# ---------------------------------------------------------------------
def _build_core(in_shape,
                filters,
                classes,
                use_up,
                axis=-1):
    inp = K.layers.Input(in_shape, name='core_in')
    p   = dict(kernel_size=(3,3,3), padding='same',
               activation=None, kernel_initializer='he_uniform')
    pT  = dict(kernel_size=(2,2,2), strides=(2,2,2),
               padding='same', kernel_initializer='he_uniform')

    # Encoder
    eA = _conv_block(inp, "eA", filters,     p); pA = K.layers.MaxPooling3D((2,2,2))(eA)
    eB = _conv_block(pA,  "eB", filters*2,   p); pB = K.layers.MaxPooling3D((2,2,2))(eB)
    eC = _conv_block(pB,  "eC", filters*4,   p); pC = K.layers.MaxPooling3D((2,2,2))(eC)
    eD = _conv_block(pC,  "eD", filters*8,   p); pD = K.layers.MaxPooling3D((2,2,2))(eD)
    eE = _conv_block(pD,  "eE", filters*16,  p)

    # Decoder
    if use_up:
        up = K.layers.UpSampling3D((2,2,2))(eE)
        # Добавляем 1x1x1 conv для приведения к нужному количеству каналов
        up = K.layers.Conv3D(filters*8, (1,1,1), padding='same', name='up_conv_D')(up)
    else:
        up = K.layers.Conv3DTranspose(filters*8, **pT, name='up_transpose_D')(eE)
    cD = CropAndConcat(axis)([up, eD])
    dC = _conv_block(cD, "dC", filters*8, p)

    if use_up:
        up = K.layers.UpSampling3D((2,2,2))(dC)
        up = K.layers.Conv3D(filters*4, (1,1,1), padding='same', name='up_conv_C')(up)
    else:
        up = K.layers.Conv3DTranspose(filters*4, **pT, name='up_transpose_C')(dC)
    cC = CropAndConcat(axis)([up, eC])
    dB = _conv_block(cC, "dB", filters*4, p)

    if use_up:
        up = K.layers.UpSampling3D((2,2,2))(dB)
        up = K.layers.Conv3D(filters*2, (1,1,1), padding='same', name='up_conv_B')(up)
    else:
        up = K.layers.Conv3DTranspose(filters*2, **pT, name='up_transpose_B')(dB)
    cB = CropAndConcat(axis)([up, eB])
    dA = _conv_block(cB, "dA", filters*2, p)

    if use_up:
        up = K.layers.UpSampling3D((2,2,2))(dA)
        up = K.layers.Conv3D(filters, (1,1,1), padding='same', name='up_conv_A')(up)
    else:
        up = K.layers.Conv3DTranspose(filters, **pT, name='up_transpose_A')(dA)
    cA = CropAndConcat(axis)([up, eA])

    out = _conv_block(cA, "out", filters, p)
    pred = K.layers.Conv3D(classes, (1,1,1),
                           activation='sigmoid',
                           name='core_pred')(out)
    return inp, pred


# ---------------------------------------------------------------------
# 6.  DVI-модель
# ---------------------------------------------------------------------
def unet_3d(input_dim,
            filters=args.filters,
            number_output_classes=args.number_output_classes,
            use_upsampling=args.use_upsampling,
            concat_axis=-1,
            model_name="DVI_UNet3D"):

    # 6.1 единый U-Net-core (shared)
    core_in, core_out = _build_core(input_dim, filters,
                                    number_output_classes,
                                    use_upsampling, concat_axis)
    core = K.models.Model(core_in, core_out, name="shared_core")

    # 6.2 I/O
    vol_in = K.layers.Input(shape=input_dim, name="MRI_vol")

    # branch α (оригинал)
    p_alpha = core(vol_in)

    # branch β (45°)
    vol_rot = K.layers.Lambda(lambda v: _rotate_vol(v, pad_val=1.0),
                              name="rot45")(vol_in)
    p_beta  = core(vol_rot)
    p_beta  = K.layers.Lambda(lambda v: _inv_rotate_vol(v),
                              name="inv_rot45")(p_beta)

    # итог — усреднение прогнозов
    pred = K.layers.Average(name="avg")([p_alpha, p_beta])

    model = K.models.Model(vol_in, pred, name=model_name)
    if args.print_model:
        model.summary()
    return model
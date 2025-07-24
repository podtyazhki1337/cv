#model
from argparser import args
import tensorflow as tf
from tensorflow import keras as K


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


def soft_dice_coef(target, prediction, axis=(1, 2, 3), smooth=0.0001):
    """
    Sorenson (Soft) Dice - Don't round predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_loss(target, prediction, axis=(1, 2, 3), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss

def ResidualBlock(x,
                  name: str,
                  filters: int,
                  params: dict,
                  dropout_rate: float = 0.20):
    """
    Two-conv residual unit, BatchNorm + ReLU, optional SpatialDropout3D.
    Сохраняет форму тензора; при смене числа каналов shortcut проецируется 1×1×1-свёрткой.
    """
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = K.layers.Conv3D(filters,
                                   kernel_size=(1, 1, 1),
                                   padding="same",
                                   kernel_initializer="he_uniform",
                                   name=f"{name}_proj")(shortcut)
        shortcut = K.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    out = K.layers.Conv3D(filters=filters, **params, name=f"{name}_conv1")(x)
    out = K.layers.BatchNormalization(name=f"{name}_bn1")(out)
    out = K.layers.Activation("relu", name=f"{name}_relu1")(out)
    if dropout_rate > 0.0:
        out = K.layers.SpatialDropout3D(dropout_rate,
                                        name=f"{name}_drop1")(out)

    out = K.layers.Conv3D(filters=filters, **params, name=f"{name}_conv2")(out)
    out = K.layers.BatchNormalization(name=f"{name}_bn2")(out)
    if dropout_rate > 0.0:
        out = K.layers.SpatialDropout3D(dropout_rate,
                                        name=f"{name}_drop2")(out)

    out = K.layers.Add(name=f"{name}_add")([shortcut, out])
    out = K.layers.Activation("relu", name=name)(out)
    return out


# -----------------------------------------------------------------------------
# 2.  SE-внимание (опционально, но хорошо подавляет FP на фоне)
#    Если не нужно — просто не вызывайте.
# -----------------------------------------------------------------------------
def SEGate(x, name: str, r: int = 8):
    """
    Squeeze-and-Excitation attention для 3-D тензора.
    """
    ch = x.shape[-1]
    z = K.layers.GlobalAveragePooling3D(name=f"{name}_gap")(x)
    z = K.layers.Dense(ch // r, activation="relu",
                       name=f"{name}_fc1")(z)
    z = K.layers.Dense(ch, activation="sigmoid",
                       name=f"{name}_fc2")(z)
    z = K.layers.Reshape((1, 1, 1, ch), name=f"{name}_reshape")(z)
    return K.layers.Multiply(name=f"{name}_scale")([x, z])

    
def unet_3d(input_dim,
            filters=args.filters,
            number_output_classes=args.number_output_classes,
            use_upsampling=args.use_upsampling,
            concat_axis=-1,
            model_name=args.saved_model_name,
            dropout_rate: float = 0.20):
    """
    U-Net-3D c резидуальными блоками и Dropout.
    Аргументы совпадают с прежними; добавлен dropout_rate.
    """

    def RB(x, name, f):                 # локальный алиас
        out = ResidualBlock(x, name, f, params, dropout_rate)
        # если нужно SE-внимание --> раскомментируйте одну строку
        # out = SEGate(out, name+"_se")
        return out

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3),
                  activation=None,
                  padding="same",
                  kernel_initializer="he_uniform")

    params_trans = dict(kernel_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        padding="same",
                        kernel_initializer="he_uniform")

    # -------------------- Encoder --------------------
    encodeA = RB(inputs,  "encodeA", filters)
    poolA   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolA")(encodeA)

    encodeB = RB(poolA,   "encodeB", filters*2)
    poolB   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolB")(encodeB)

    encodeC = RB(poolB,   "encodeC", filters*4)
    poolC   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolC")(encodeC)

    encodeD = RB(poolC,   "encodeD", filters*8)
    poolD   = K.layers.MaxPooling3D(pool_size=(2, 2, 2), name="poolD")(encodeD)

    encodeE = RB(poolD,   "encodeE", filters*16)

    # -------------------- Decoder --------------------
    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upE")(encodeE)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters*8, **params_trans,
                                   name="transconvE")(encodeE))
    concatD = K.layers.concatenate([up, encodeD], axis=concat_axis,
                                   name="concatD")
    decodeC = RB(concatD, "decodeC", filters*8)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upC")(decodeC)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters*4, **params_trans,
                                   name="transconvC")(decodeC))
    concatC = K.layers.concatenate([up, encodeC], axis=concat_axis,
                                   name="concatC")
    decodeB = RB(concatC, "decodeB", filters*4)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upB")(decodeB)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters*2, **params_trans,
                                   name="transconvB")(decodeB))
    concatB = K.layers.concatenate([up, encodeB], axis=concat_axis,
                                   name="concatB")
    decodeA = RB(concatB, "decodeA", filters*2)

    up = (K.layers.UpSampling3D(size=(2, 2, 2), name="upA")(decodeA)
          if use_upsampling else
          K.layers.Conv3DTranspose(filters, **params_trans,
                                   name="transconvA")(decodeA))
    concatA = K.layers.concatenate([up, encodeA], axis=concat_axis,
                                   name="concatA")

    convOut = RB(concatA, "convOut", filters)

    prediction = K.layers.Conv3D(filters=number_output_classes,
                                 kernel_size=(1, 1, 1),
                                 activation="sigmoid",
                                 name="PredictionMask")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction],
                           name=model_name)

    if args.print_model:
        model.summary()

    return model

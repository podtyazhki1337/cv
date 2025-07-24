from __future__ import print_function, unicode_literals, absolute_import, division
from comet_ml import Experiment
import sys
import math
import numpy as np
import shutil
import matplotlib
import os
from glob import glob
from pathlib import Path
import pprint
import random
import tensorflow as tf
from scipy.ndimage import gaussian_filter
#matplotlib.rcParams["image.interpolation"] = None
#import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import skimage
import imageio
from tqdm import tqdm
from tifffile import imread
from tensorflow.keras.utils import Sequence
from skimage.transform import rescale
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from skimage.transform import AffineTransform, warp
import scipy, pathlib
import config
import init
import misc
import inspect, types
from scipy.ndimage import affine_transform
from types import MethodType
experiment = Experiment(
    api_key="",
    project_name="stardist3d-rat-neurons",
    workspace="podtyazhki1337",      
    auto_output_logging="simple",    
)

pp = pprint.PrettyPrinter(indent=4)
orig_train = StarDist3D.train   # сохраним оригинал

_orig_train = StarDist3D.train           # сохраняем исходный метод

def train_with_callbacks(self, *args, callbacks=None, **kw):
    """обёртка, которая прокидывает callbacks в keras_model.fit()"""
    cb = list(callbacks or [])
    fit_orig = self.keras_model.fit

    def fit_wrapped(*fa, **fk):
        fk["callbacks"] = list(fk.get("callbacks", [])) + cb
        return fit_orig(*fa, **fk)

    self.keras_model.fit = fit_wrapped
    try:
        return _orig_train(self, *args, **kw)
    finally:
        self.keras_model.fit = fit_orig     # вернуть как было

StarDist3D.train = train_with_callbacks

def compute_dice(gt, pred):
    inter = np.logical_and(gt > 0, pred > 0).sum()
    vol   = (gt > 0).sum() + (pred > 0).sum()
    return (2*inter) / vol if vol else 1.0

class DiceCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, sd_model, val_x, val_y,
                 out_dir="dice_ckpts", eval_every=1):
        super().__init__()
        self.sd, self.vx, self.vy = sd_model, val_x, val_y
        self.out = pathlib.Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.every, self.best = eval_every, -np.inf                    # лучшее значение

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.every:          # считаем каждые eval_every эпох
            return

        dices = []
        for i in range(len(self.vx)):
            img = self.vx[i]                 # уже нормализован в StarDistSequence
            gt  = self.vy[i]
            pr, _ = self.sd.predict_instances(
                        img, axes='ZYX',
                        n_tiles=self.sd._guess_n_tiles(img),
                        show_tile_progress=False)
            dices.append(compute_dice(gt, pr))

        mean_dice = float(np.mean(dices))
        print(f"\n🔍  Epoch {epoch+1}: mean-Dice = {mean_dice:.4f}")

        if mean_dice > self.best:
            self.best = mean_dice
            src = self.sd.logdir / "weights_best.h5"        # ← правильный путь в 0.9.1
            dst = self.out / f"bestDice_ep{epoch+1:03d}_{mean_dice:.4f}.h5"
            shutil.copy(src, dst)
            print(f"   🟢  Dice improved → saved {dst}")


def preproc_data(in_dataset_dir, out_dataset_dir, validation_images=[], skip=[]):
    '''
    Preprocesses the input dataset.

    1. Checks the images and masks directory.
    2. Reads all the data.
    3. Writes the metadata to out into a json for each sample.
    4. Crops the content of the image & mask and writes to out.
    '''

    out_dir = {
        'train': Path(out_dataset_dir) / 'train',
        'val':  Path(out_dataset_dir) / 'val'
    }

    meta = {
        'train': [],
        'val': []
    }

    print('Preprocessing...')
    in_dataset_dir = Path(in_dataset_dir)
    out_dataset_dir = Path(out_dataset_dir)

    X = sorted((in_dataset_dir/'images').glob('*.tif'))
    Y = sorted((in_dataset_dir/'masks').glob('*.tif'))

    print('Images:')
    pp.pprint(X)
    print('Masks:')
    pp.pprint(Y)

    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

    def get_crop(bbox, pad, dims):
        '''
        bbox = (zmin, ymin, xmin, zmax, ymax, xmax)
        dims = (zmax_, ymax_, xmax_)    (Shape of the image)
        pad = (zpad, ypad, xpad)        (Padding of each dim)

        '''
        ndim = len(bbox)//2
        return tuple(slice(  max(0, bbox[d]-pad[d]), min(dims[d], bbox[d+ndim]+pad[d])  ) for d in range(ndim))

    def measure_cells(props):
        cells = []
        extents = []
        for prop in props:
            cell_bbox = prop.bbox
            ndim = len(cell_bbox) // 2
            cell = np.array(list([cell_bbox[d+ndim]-cell_bbox[d] for d in range(ndim)]))
            cells.append(cell)

            extent = np.array(cell_bbox[ndim:])-np.array(cell_bbox[:ndim])
            extents.append(extent)
        
        return np.median(np.array(cells), axis=0), np.mean(np.array(cells), axis=0), np.median(extents, axis=0)

    for x, y in zip(X, Y):
        x_im = imageio.volread(x)
        y_im = imageio.volread(y)

        sample_basename = os.path.basename(x)

        if sample_basename in skip:
            continue

        if sample_basename in validation_images:
            subset = 'val'
        else:
            subset = 'train'

        print(sample_basename, '->', subset)

        props = skimage.measure.regionprops((y_im > 0).astype(np.uint8))
        bbox = props[0].bbox
        crop = get_crop(bbox, [config.padsize]*3, np.shape(y_im))

        cell_sizes_median_o, cell_sizes_mean_o, cell_extent_o = measure_cells(skimage.measure.regionprops(y_im))

        processed_x = x_im[crop]
        processed_y = y_im[crop]

        # ...........
        print(y, y)

        rescale_factor = np.array(config.std_cell_size) / cell_sizes_mean_o
        print(processed_x.dtype, processed_y.dtype)

        if False:
            plt.subplot(121)
            plt.imshow(processed_x[len(processed_x)//2, ...])
            plt.subplot(122)
            plt.imshow(processed_y[len(processed_y)//2, ...])
            plt.show()

        processed_x = rescale(processed_x, rescale_factor, order=1, anti_aliasing=False, preserve_range=True).astype(np.uint16)
        processed_y = rescale(processed_y, rescale_factor, order=0, anti_aliasing=False, preserve_range=True).astype(np.uint16)
        processed_y = fill_label_holes(processed_y)

        if False:
            plt.subplot(121)
            plt.imshow(processed_x[len(processed_x)//2, ...])
            plt.subplot(122)
            plt.imshow(processed_y[len(processed_y)//2, ...])
            plt.show()

        cell_sizes_median_p, cell_sizes_mean_p, cell_extent_p = measure_cells(skimage.measure.regionprops(processed_y))

        #axis_norm = (0,1,2)
        #processed_x = normalize(processed_x, 1, 99.8, axis=axis_norm)
        # ------------

        ################

        # x * x' = x*
        # x' = x*/x
        #std_transform = np.array(config.std_cell_size) / cell_sizes_mean

        # Put JSON
        sample = {
            'name': sample_basename,
            
            'x_dims': x_im.shape,
            'x_dtype': str(x_im.dtype),
            'x_limits': (int(np.min(x_im)), int(np.max(x_im))),
            
            'y_dims': y_im.shape,
            'y_dtype': str(y_im.dtype),
            'y_limits': (int(np.min(y_im)), int(np.max(y_im))),
            'y_unique_values': len(np.unique(y_im)),
            
            'cells_median_o': tuple(cell_sizes_median_o.tolist()),
            'cells_mean_o': tuple(cell_sizes_mean_o.tolist()),
            'cells_median_extent_o': tuple(cell_extent_o.tolist()),

            'cells_median_p': tuple(cell_sizes_median_p.tolist()),
            'cells_mean_p': tuple(cell_sizes_mean_p.tolist()),
            'cells_median_extent_p': tuple(cell_extent_p.tolist()),

            'x_dims_p': np.shape(processed_x)
        }
        
        meta[subset].append(sample)

        # Print CSV
        print('%s;%s;%s;%d;%d;%s;%s;%d;%d;%d;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f' % (
            sample_basename, 
            x_im.shape, 
            x_im.dtype, 
            np.min(x_im), 
            np.max(x_im), 
            
            y_im.shape, 
            y_im.dtype,
            len(np.unique(y_im)), 
            np.min(y_im), 
            np.max(y_im),
            *tuple(cell_sizes_median_o),
            *tuple(cell_sizes_mean_o)))

        # Copy the processed to the new path
        (out_dir[subset]/'images').mkdir(exist_ok=True, parents=True)
        (out_dir[subset]/'masks').mkdir(exist_ok=True, parents=True)

        imageio.volwrite(out_dir[subset]/'images'/sample_basename, processed_x)
        imageio.volwrite(out_dir[subset]/'masks'/sample_basename, processed_y)

    for subset_ in ['train', 'val']:
        es = []
        for s in meta[subset_]:
            es.append(s['cells_median_extent_p'])

        print('Extents (%s):' % subset_, np.median(np.stack(es), axis=0))

        misc.put_json(out_dir[subset_] / 'meta.json', meta[subset_])
    
    print('Done.')

class StarDistSequence(Sequence):
    '''
    Loads the data into the memory efficiently on-demand.
    It first reads the metadata of the processed database and stores it.
    Then, a random subset of images loaded in a fixed frequency.
    '''

    def __init__(self, data_dir, y=False, pool_size=None, repool_freq=None):
        '''
        @arg pool_size: how many images should be stored at a time in the memory.
            -1: all the data.
        @arg repool_frequency: how often the dataset should be reloaded.
        '''
        self.y = y
        if self.y == True:
            subdir = 'masks'
        else:
            subdir = 'images'
        
        self.data_dir = Path(data_dir) / subdir
        self.meta = misc.get_json(Path(data_dir)/'meta.json')
        self.counter = 0
        self.r = random.Random(42)

        assert pool_size !=0 and repool_freq != 0, "Pool_size and repool_freq should not be 0!"

        if pool_size is None:
            self.pool_size = len(self.meta)
        else:
            self.pool_size = pool_size
        
        if repool_freq is None:
            self.repool_freq = self.pool_size
        else:
            self.repool_freq = self.pool_size*repool_freq
        
        self.data = None

        print('Database:')
        for idx,im in enumerate(self.meta):
            print(idx, '\t', im['name'])
        
        self.repool()

    def get_repool_frequency(self):
        return self.repool_freq

    def get_cell_properties(self, prop='cells_median_extent_p'):
        '''
        Returns the image properties as numpy array.
        '''
        extents = []
        for s in self.meta:
            extents.append(s[prop])
        return np.array(extents)

    def on_epoch_end(self):
        pass

    def get_real_index(self, k):
        return self.new_pool_ids[k]

    def repool(self, seed=0):
        print('Repooling...')
        
        r_ = random.Random(seed)

        sample_ids = list(range(len(self.meta)))
        self.new_pool_ids = r_.sample(sample_ids, self.pool_size)
        self.pool = [self.meta[i] for i in self.new_pool_ids]

        self.data = []
        for im in self.pool:
            image = imageio.volread(self.data_dir / im['name'])
            #rescale_factor = np.array(config.std_cell_size) / im['cells_mean']
            #print('Image:', im['name'])
            #print('STD cell size:', np.array(config.std_cell_size))
            #print('Image mean cell size:', im['cells_mean'])
            #print('Rescale factor:', rescale_factor)

            #o = 0 if self.y == True else 1
            #rescale_factor = np.array(config.std_cell_size) / im['cells_mean']
            #image = rescale(image, rescale_factor, order=o, preserve_range=True)

            if self.y == True:
                image = image.astype(np.uint16)
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = self.data[idx].copy()

        if self.y == False:
            from csbdeep.utils import Path, normalize
            axis_norm = (0,1,2)

            ret = normalize(ret.astype(np.float32), 1,99.8, axis=axis_norm)

        return ret

def load_data():
    '''
    Loads the data eagerly. This code is mainly from the 3D example.
    '''

    X = sorted(glob('/images/*.tif'))
    Y = sorted(glob('/masks/*.tif'))
    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

    def lazy_arr(arr):
        for a in arr:
            yield imread(a)

    # Lazy laods the images
    def lazy_sample(X, Y, im_idx=None):
        if im_idx == None:
            for x, y in zip(X, Y):
                yield imread(x), imread(y)
        else:
            yield imread(X[im_idx]), imread(Y[im_idx])

    print('Dataset:')
    dataset = lazy_sample(X, Y)

    for idx, (x, y) in enumerate(dataset):
        print(idx, np.shape(x), np.shape(y))

    dataset_first = lazy_sample(X, Y, 0)
    X0, Y0 = list(dataset_first)[0]

    n_channel = 1 if X0.ndim == 3 else X0.shape[-1]

    Y_all_1 = lazy_arr(Y)
    Y_all_2 = lazy_arr(Y)

    X = list(map(imread,X))
    Y = list(map(imread,Y))

    print('Channels:', n_channel)

    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    extents = calculate_extents(Y_all_1)
    print('EXTENTS:', extents)
    anisotropy = tuple(np.max(extents) / extents)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

    median_size = calculate_extents(Y_all_2, np.median)

    return X, Y, X_trn, Y_trn, X_val, Y_val, n_channel, anisotropy, median_size

def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

def show_example(X, Y, i):
    print('Plotting:', i, np.shape(X), np.shape(Y))
    img, lbl = X[i], Y[i]
    assert img.ndim in (3,4)
    img = img if img.ndim==3 else img[...,:3]
    plot_img_label(img,lbl)
    plt.show()

def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img
def zoom_3d_center(vol, zoom, order):
    """
    Центрированный зум 3D-стека:
     - zoom>1: увеличение, zoom<1: уменьшение
     - order=1 для изображений, 0 для меток
    """
    # matrix for mapping output coords -> input coords
    # (делим по Y,X на zoom, Z остаётся без изменений)
    matrix = np.diag([1, 1/zoom, 1/zoom])
    # центр исходного объёма
    center = (np.array(vol.shape) - 1) / 2.0
    # чтобы масштабировалось вокруг центра
    offset = center - matrix.dot(center)
    return affine_transform(
        vol, 
        matrix=matrix,
        offset=offset,
        order=order,
        mode='constant', 
        cval=0
    )
def augmenter(x, y):
    # flip/rot + intensity
    x, y = random_fliprot(x, y, axis=(1,2))
    x = x * np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)

    # --- честный 3D-zoom ±20% вокруг центра ---
    # zoom = np.random.uniform(0.9, 1.1)
    # x = zoom_3d_center(x, zoom, order=1)
    # y = zoom_3d_center(y, zoom, order=0)
    
    # --- лёгкий 3D Gaussian blur ---
    # if np.random.rand() < 0.5:
    #     sigma = np.random.uniform(0.3, 0.6)  # очень слабый
    #     x = gaussian_filter(x, sigma=sigma)
        
    # --- gamma 0.7–1.3 ---
    gamma = np.random.uniform(0.9, 1.1)
    x = np.clip(x, 0, None)
    x = x ** gamma

    noise = np.random.normal(loc=0.0, scale=0.01, size=x.shape)
    x = x + noise
    x = np.clip(x, 0, None)

    return x, y


def show_aug_example(X, Y, i):
    # plot some augmented examples
    img, lbl = X[0],Y[0]
    plot_img_label(img, lbl)
    for _ in range(3):
        img_aug, lbl_aug = augmenter(img,lbl)
        plot_img_label(img_aug, lbl_aug, img_title="image augmented (XY slice)", lbl_title="label augmented (XY slice)")

if __name__ == '__main__':
    
    np.random.seed(42)
    
    preproc = False

    if preproc:
        #Every image goes to the train, except the validation images.
        preproc_data(config.in_dataset_dir, config.processed_dataset_dir, config.validation_images, config.skip)
    else:
        lbl_cmap = random_label_cmap()

        def get_dataset(dataset_dir, pool_size=None, repool_freq=None):
            return StarDistSequence(dataset_dir, pool_size=pool_size, repool_freq=repool_freq), StarDistSequence(dataset_dir, y=True, pool_size=pool_size, repool_freq=repool_freq)
        
        trn_seq_x, trn_seq_y = get_dataset(Path(config.processed_dataset_dir) / 'train', config.pool_size, config.repool_freq)
    
        val_seq_x, val_seq_y = get_dataset(Path(config.processed_dataset_dir) / 'val')

        train_medians = trn_seq_x.get_cell_properties()
        train_dataset_median = np.median(train_medians, axis=0)

        print('Train dataset median cell size:', train_dataset_median)

        extents = train_dataset_median
        anisotropy = tuple(np.max(extents) / extents)

        median_size = extents

        n_channel = 1

        print(Config3D.__doc__)
    
        n_rays = 128

        quick_demo = False

        # Use OpenCL-based computations for data generator during training (requires 'gputools')
        use_gpu = False and gputools_available()
        # use_gpu = True
        # Predict on subsampled grid for increased efficiency and larger field of view
        grid = tuple(1 if a > 1.5 else 4 for a in anisotropy)

        # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
        rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

        conf = Config3D (
            rays             = rays,
            grid             = grid,
            anisotropy       = anisotropy,
            use_gpu          = use_gpu,
            n_channel_in     = n_channel,
            # adjust for your data below (make patch size as large as possible)
            train_patch_size = (12,96,96),
            train_batch_size = 4,
            
        )
    
        print(conf)
        vars(conf)
        
        experiment.log_parameters({
        "train_patch_size": conf.train_patch_size,
        "train_batch_size": conf.train_batch_size,
        "n_rays": n_rays,
        "grid": grid,
        "anisotropy": anisotropy,
        "zoom_aug_range": [0.8, 1.2],
        "gamma_aug_range": [0.7, 1.3],})
        
        if use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory
            # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
            #limit_gpu_memory(0.8)
            # alternatively, try this:
            limit_gpu_memory(None, allow_growth=True)

        model = StarDist3D(conf, name='stardist_hela', basedir='models')
        experiment.set_model_graph(str(conf))
        dice_cb = DiceCheckpoint(model, val_seq_x, val_seq_y,
                    out_dir="models/stardist_hela/stardist_dice_ckpts", eval_every=5)
        
        

        
        fov = np.array(model._axes_tile_overlap('ZYX'))
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")
        schedule = ExponentialDecay(initial_learning_rate=3e-4,decay_steps=700,decay_rate=0.5,staircase=True)
        opt = Adam(learning_rate=schedule)
        
        if quick_demo:
            print (
                "NOTE: This is only for a quick demonstration!\n"
                "      Please set the variable 'quick_demo = False' for proper (long) training.",
                file=sys.stderr, flush=True
            )
            '''
            model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                        epochs=2, steps_per_epoch=5)
            '''

            print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
            model = StarDist3D.from_pretrained('3D_demo')
        else:
            steps = trn_seq_x.repool_freq
            model.keras_model.optimizer = opt
            model.train(trn_seq_x, trn_seq_y, validation_data=(val_seq_x, val_seq_y), augmenter=augmenter, epochs=400, steps_per_epoch=steps,  callbacks       = [dice_cb])
           
            experiment.log_metric("train_steps_per_epoch", steps)
        thr = model.optimize_thresholds(val_seq_x, val_seq_y)
        experiment.log_metrics({"optimized_prob_thresh": thr})

        
        # To make sure that I see fov at the end
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")


        # Validation


        Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                              for x in tqdm(val_seq_x)]
        inter_total, union_total = 0, 0
        for i, pred in enumerate(Y_val_pred):
            gt = val_seq_y[i]
            inter_total += np.logical_and(gt > 0, pred > 0).sum()
            union_total += np.logical_or(gt > 0, pred > 0).sum()
        global_iou = inter_total / union_total if union_total else 0.0
        print(f"Global 3D IoU across dataset: {global_iou:.4f}")
        experiment.log_metric("global_3d_iou", global_iou)
        
        #save_tiff_imagej_compatible('pred0_image.tif', Y_val_pred[0], axes='ZYX')
        #save_tiff_imagej_compatible('pred1_image.tif', Y_val_pred[1], axes='ZYX')

        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(val_seq_y, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
        print(stats)
        #experiment.log_model(model.name, model.basedir)

        experiment.end()
      #  stats[taus.index(0.1)]
      #  stats[taus.index(0.2)]
      #  stats[taus.index(0.3)]
      #  stats[taus.index(0.4)]
      #  stats[taus.index(0.5)]
      #  stats[taus.index(0.6)]
      #   stats[taus.index(0.7)]
      #  stats[taus.index(0.8)]
      #  stats[taus.index(0.9)]




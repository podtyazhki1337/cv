import config

config.pool_size = None
config.repool_freq = 10



# Resize below the FOV (17, 30, 30)
#config.std_cell_size = (5, 27, 27)
config.std_cell_size = (5, 28, 28)

#config.in_dataset_dir = '/storage01/miroslavm/3d_segmentation_models/data_rat'
config.in_dataset_dir ='/NAS/mmaiurov/Datasets/Hela_MRC/'
# config.processed_dataset_dir = '/storage01/miroslavm/3d_segmentation_models/Stardist/processed_w_stardist/'
config.processed_dataset_dir = '/NAS/mmaiurov/Datasets/Hela_MRC_stardist/'
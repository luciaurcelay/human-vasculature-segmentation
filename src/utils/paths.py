import os


# Get the path from raw images
def get_img_path():
    ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_PATH = os.path.join(ABS_PATH, 'data/raw/hubmap-hacking-the-human-vasculature')
    IMG_PATH = os.path.join(DATA_PATH, 'train/')

    return IMG_PATH


# Get the path from metadata files
def get_metadata_paths():
    ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_PATH = os.path.join(ABS_PATH, 'data/raw/hubmap-hacking-the-human-vasculature')
    POLYGONS_PATH = os.path.join(DATA_PATH, 'polygons.jsonl')
    WSI_META_PATH = os.path.join(DATA_PATH, 'wsi_,eta.csv')
    TILE_META_PATH = os.path.join(DATA_PATH, 'tile_meta.csv')

    return POLYGONS_PATH, WSI_META_PATH, TILE_META_PATH


# Create custom data directories
def create_custom_data_directory():
    ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print(ABS_PATH)
    DATA_PATH = os.path.join(ABS_PATH, 'data/custom')
    print("Created 'custom data' directory: ", DATA_PATH)

    TRAIN_IMG = os.path.join(DATA_PATH, 'train/image/')
    TRAIN_MASK = os.path.join(DATA_PATH, 'train/mask/')
    TEST_IMG = os.path.join(DATA_PATH, 'test/image/')
    TEST_MASK = os.path.join(DATA_PATH, 'test/mask/')

    os.makedirs(TRAIN_IMG, exist_ok=True)
    os.makedirs(TRAIN_MASK, exist_ok=True)
    os.makedirs(TEST_IMG, exist_ok=True)
    os.makedirs(TEST_MASK, exist_ok=True)

    return TRAIN_IMG, TRAIN_MASK, TEST_IMG, TEST_MASK
    
    
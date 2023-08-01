import os


# Get the path from raw images
def get_img_path():

    ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_PATH = os.path.join(ABS_PATH, 'data/raw/hubmap-hacking-the-human-vasculature')
    IMG_PATH = os.path.join(DATA_PATH, 'train')

    return IMG_PATH


# Get the path from metadata files
def get_metadata_path():

    ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_PATH = os.path.join(ABS_PATH, 'data/raw/hubmap-hacking-the-human-vasculature')
    POLYGONS_PATH = os.path.join(DATA_PATH, 'polygons.jsonl')
    WSI_META_PATH = os.path.join(DATA_PATH, 'wsi_,eta.csv')
    TILE_META_PATH = os.path.join(DATA_PATH, 'tile_meta.csv')

    return POLYGONS_PATH, WSI_META_PATH, TILE_META_PATH
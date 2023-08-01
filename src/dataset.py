from utils.paths import get_img_path, get_metadata_path

def create_dataset():

    IMG_PATH = get_img_path()

    POLYGONS_PATH, WSI_META_PATH, TILE_META_PATH = get_metadata_path()

    print(IMG_PATH, TILE_META_PATH)
    
    return None



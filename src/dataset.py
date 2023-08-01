from utils.paths import get_img_path, get_metadata_paths, create_custom_data_directory

import pandas as pd
import json
import tifffile as tiff
from tqdm import tqdm
import cv2
import imagecodecs
from PIL import Image
import numpy as np
import os

def create_dataset():
    # Get data paths
    IMG_PATH = get_img_path()
    POLYGONS_PATH, WSI_META_PATH, TILE_META_PATH = get_metadata_paths()

    # Create directory for custom data
    TRAIN_IMG, TRAIN_MASK, TEST_IMG, TEST_MASK = create_custom_data_directory()

    data_gen = len(os.listdir(TRAIN_IMG))

    if data_gen > 0:
        print("Custom dataset has already been generated.")

    else:
        print("Generating custom dataset.")

        # Create dict containing tile information
        tiles_dicts, tile_meta_df = create_tiles(POLYGONS_PATH, TILE_META_PATH)

        # Create new dataset containing only human vasculature masks
        for i, tldc in enumerate(tqdm(tiles_dicts)):
            tile_id = tldc["id"]
            # Find the corresponding row in the DataFrame where 'id' matches the filename
            row = tile_meta_df[tile_meta_df['id'] == tile_id]
            # Check the 'dataset' value for the identified row
            dataset_value = row['dataset'].values[0]
            array = tiff.imread(f'{IMG_PATH}{tldc["id"]}.tif')
            img_example = Image.fromarray(array)
            img = np.array(img_example)
            mask = make_seg_mask(tldc)
            
            if np.sum(mask)>0:
                
                if dataset_value == 1:

                    cv2.imwrite(f'{TEST_IMG}{tldc["id"]}.png', img)
                    cv2.imwrite(f'{TEST_MASK}{tldc["id"]}_mask.png', mask)
                    
                else:
                    
                    cv2.imwrite(f'{TRAIN_IMG}{tldc["id"]}.png', img)
                    cv2.imwrite(f'{TRAIN_MASK}{tldc["id"]}_mask.png', mask)


    return None


# Create tiles from dataset
def create_tiles(POLYGONS_PATH, TILE_META_PATH):
    with open(POLYGONS_PATH, "r") as json_file:
        json_list = list(json_file)

    tiles_dicts = []
    for json_elem in json_list:
        tiles_dicts.append(json.loads(json_elem))

    tile_meta_df = pd.read_csv(TILE_META_PATH)

    return tiles_dicts, tile_meta_df


# Create masks of blood vessels
def make_seg_mask(tiles_dict):
    mask = np.zeros((512, 512), dtype=np.float32)
    for annot in tiles_dict['annotations']:
        cords = annot['coordinates']
        if annot['type'] == "blood_vessel":
            for cd in cords:
                rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
                mask[rr, cc] = 1
                
    contours,_ = cv2.findContours((mask*255).astype(np.uint8), 1, 2)
    zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")

    for p in contours:
        cv2.fillPoly(zero_img, [p], (255, 255, 255))

    contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_with_area = zero_img

    for i in range(len(contours)):
        cv2.fillPoly(img_with_area, [contours[i][:,0,:]], (255-4*(i+1),255-4*(i+1),255-4*(i+1)), lineType=cv2.LINE_8, shift=0)
            
    return img_with_area 


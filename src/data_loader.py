from utils.datasets import HubMapDataset
from utils.config import CFG
from utils.path import get_custom_img_paths

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import gc


def create_dataloaders(TRAIN_IMG_PATH, TRAIN_MASK_PATH, TEST_IMG_PATH, TEST_MASK_PATH):
    # Clean cache
    gc.collect()
    torch.cuda.empty_cache()

    # Get data paths
    TRAIN_IMG, TRAIN_MASK, TEST_IMG, TEST_MASK = get_custom_img_paths()

    # Create Datasets
    train_dataset = HubMapDataset(
        image_dir=TRAIN_IMG, 
        mask_dir=TRAIN_MASK, 
        img_transform=CFG.image_transform, 
        mask_transform=CFG.mask_transform
    )

    test_dataset = HubMapDataset(
        image_dir=TEST_IMG, 
        mask_dir=TEST_MASK, 
        img_transform=CFG.image_transform, 
        mask_transform=CFG.mask_transform
    )

    # Create train validation split
    train_dataset, val_dataset = random_split(train_dataset, [0.6, 0.4])

    print(f'Train split length: {len(train_dataset)}')
    print(f'Val split length: {len(val_dataset)}')
    print(f'Test split length: {len(test_dataset)}')


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size_train, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size_val, shuffle=False)

    return train_loader, val_loader, test_loader


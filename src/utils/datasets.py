import os
from torch.utils.data import Dataset


class HubMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

        
    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Open image and mask using PIL (Python Imaging Library)
        image = Image.open(image_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)
        
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
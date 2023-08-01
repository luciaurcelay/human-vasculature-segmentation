from segmentation_models_pytorch import utils
import segmentation_models_pytorch as smp
from torchvision import transforms


# Config class
class CFG:
    
    batch_size_train = 6
    batch_size_val = 4
    size = 512
    org_size = 512
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
      torch.cuda.manual_seed(1)
    
    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True
    
    # Set num of epochs
    EPOCHS = 16
    
    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]


    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize((size, size)),  # Resize the image to the desired size
        transforms.ToTensor(), # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize image
    ])

    # Mask transformations (no normalization applied)
    mask_transform = transforms.Compose([
        transforms.Resize((size, size)),  # Resize the mask to the desired size
        transforms.ToTensor(), # Convert PIL image to PyTorch tensor
    ])
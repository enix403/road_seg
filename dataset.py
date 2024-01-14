from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm 
from torchvision.transforms import v2

ROOT_DIR = Path("data/globe-data")

preprocess = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.CenterCrop(256),
    v2.Resize((256, 256))
])

def load_single(image_path: str, mask_path: str):
    img = Image.open(image_path)
    mask = Image.open(mask_path)

    # (3, 256, 256)
    img = preprocess(img)

    # (3, 256, 256)
    mask = preprocess(mask)

    # (1, 256, 256)
    mask = mask[:1]

    return img, mask


def load_all():

    # indices = torch.randperm(6226)[:512]
    # torch.save([X_all[indices], Y_all[indices]], "data/small.pt")
    # torch.save([X_all, Y_all], "data/small.pt")

    X_all, Y_all = torch.load('data/small.pt', weights_only=True)
    return X_all, Y_all

    # --------------------

    data = torch.load('data/quick.pt', weights_only=True)
    X_all = data['X_all']
    Y_all = data['Y_all']

    return X_all, Y_all

    # --------------------

    metadata = pd.read_csv(ROOT_DIR / "metadata.csv")
    metadata = metadata[metadata["split"] == "train"]

    all_images = []
    all_masks = [] 

    for image_path, mask_path in tqdm(zip(metadata['sat_image_path'], metadata['mask_path'])):
        img, mask = load_single(image_path, mask_path)
        all_images.append(img)
        all_masks.append(mask)


    X_all = torch.stack(all_images, dim=0)
    Y_all = torch.stack(all_masks, dim=0)

    torch.save({ 'X_all': X_all, 'Y_all': Y_all }, 'data/quick.pt')

    return X_all, Y_all
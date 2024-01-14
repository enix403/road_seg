from PIL import Image

_image_path = "train/100034_sat.jpg"
_mask_path = "train/100034_mask.png"

def load_single(image_path: str, mask_path: str):
    pass



preprocess = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(227),
    v2.Resize((256, 256))
])


def show_pair(img, mask, p=False):
    if p:
        img = preprocess(img)
        mask = preprocess(mask)

        img = v2.functional.to_pil_image(img)
        mask = v2.functional.to_pil_image(mask)

        # isinstance(torch.randn(1,2), Image.Image)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)




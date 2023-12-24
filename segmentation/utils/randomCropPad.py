from PIL import Image, ImageOps
from torchvision import transforms


#========================== Pad the image for RandomResizedCrop ==============================
class RandomCropAndPad:
    def __init__(self, size, scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        aspect_ratio = img.width / img.height

        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, (aspect_ratio, aspect_ratio))
        img = transforms.functional.crop(img, i, j, h, w)

        # Calculate padding
        delta_width = self.size - w
        delta_height = self.size - h
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

        img = ImageOps.expand(img, padding)

        img = transforms.functional.resize(img, (self.size, self.size))

        return img

class RandomCropAndPadMask:
    def __init__(self, size, scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, mask):
        aspect_ratio = mask.width / mask.height

        i, j, h, w = transforms.RandomResizedCrop.get_params(mask, self.scale, (aspect_ratio, aspect_ratio))
        mask = transforms.functional.crop(mask, i, j, h, w)

        delta_width = self.size - w
        delta_height = self.size - h
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

        mask = ImageOps.expand(mask, padding)

        mask = transforms.functional.resize(mask, (self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST)

        return mask
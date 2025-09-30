# src/transforms.py (修正為 v1 API)
import torchvision.transforms as T


# 為了手動處理 bounding box 的轉換，我們需要一個輔助類
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# 這個 transform 只會作用在 image 上
class ToTensor:
    def __call__(self, image, target):
        return T.ToTensor()(image), target


# 這個 transform 會同時作用在 image 和 target 上
class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        import random

        import torchvision.transforms.functional as F

        if random.random() < self.prob:
            image = F.hflip(image)
            # 我們也必須手動翻轉 bounding box！
            width, _ = image.size
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

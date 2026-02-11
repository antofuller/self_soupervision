import torch
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation
import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandAugment,
    RandomHorizontalFlip,
)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class GaussianBlur(object):
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
     
class horizontal_flip(object):
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
class ResizeSmall(object):
    def __init__(self, smaller_size):
        assert isinstance(smaller_size, (int))
        self.smaller_size = smaller_size

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]

        ratio = float(self.smaller_size) / min(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)

        image = transforms.Resize((new_h, new_w), antialias=True)(image)
        return image
    
def new_data_aug_generator(simple_random_crop,color_jitter,img_size):
    remove_random_resized_crop = simple_random_crop
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                            Solarization(p=1.0),
                                            GaussianBlur(p=1.0)])]
    if color_jitter is not None and not color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))

    return primary_tfl+secondary_tfl


def get_3aug():
    first_tfl = new_data_aug_generator(simple_random_crop=False, color_jitter=0.3, img_size=224)
    final_tfl = [
        ToTensor(),
        Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    transform = Compose(first_tfl + final_tfl)
    return transform


def get_randaug():
    first_tfl = [
        RandomResizedCropAndInterpolation(
            224, scale=(0.08, 1.0), interpolation="bicubic"
        ),
        RandomHorizontalFlip(),
        RandAugment(2, 15),
    ]

    final_tfl = [
        ToTensor(),
        Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    transform = Compose(first_tfl + final_tfl)
    return transform



class miniVTAB(Dataset):
    def __init__(self, dataset_name, split, augmentation, do_two_views=False):
        assert augmentation in ["3aug", "randaug", "none"]
        self.dataset = load_dataset("antofuller/mini-VTAB", dataset_name, split=split)
        self.do_two_views = do_two_views

        if augmentation == "none":
            self.transform = Compose(
                [
                    ToTensor(),
                    Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                ]
            )

        elif augmentation == "3aug":
            self.transform = get_3aug()

        elif augmentation == "randaug":
            self.transform = get_randaug()

    def __len__(self):
        return len(self.dataset)

    def get_num_classes(self):
        labels = [example["label"] for example in self.dataset]
        return max(labels) + 1

    def __getitem__(self, idx):
        original_image = self.dataset[idx]["image"].convert("RGB")
        label = self.dataset[idx]["label"]
        if self.do_two_views:
            # just for two-view SSL methods
            image_1 = self.transform(original_image)
            image_2 = self.transform(original_image)
            return image_1, image_2, label
        else:
            image = self.transform(original_image)
            return image, label


if __name__ == "__main__":
    # just to test some stuff eh
    dataset = miniVTAB(
        dataset_name="cifar100",
        split="train",
        augmentation="none",
        do_two_views=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    for batch in loader:
        images, labels = batch
        print(images.shape)  # should be (256, 3, 224, 224)
        print(images[:, 0, :, :].mean(), images[:, 0, :, :].std())  # should be ~0 and ~1
        print(images[:, 1, :, :].mean(), images[:, 1, :, :].std())  # should be ~0 and ~1
        print(images[:, 2, :, :].mean(), images[:, 2, :, :].std())  # should be ~0 and ~1
        exit()
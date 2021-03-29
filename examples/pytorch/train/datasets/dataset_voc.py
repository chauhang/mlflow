import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as transforms

root = r"C:\Users\Windows\Facebook_Eng2\GC_Repos\mlflow-torchserve-1\examples\Semantic_Segmentation_E2E\datasets\VOC2012"


class VOC(Dataset):

    voc_colormap = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )

    def __init__(self, is_train, crop_size=(320, 480)):
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.imgs, self.masks = self.get_voc_images(root, is_train)
        self.imgs, self.masks = self.imgs, self.masks
        print(len(self.imgs))
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images, please check the data set")

        self.imgs = [self.normalize_image(img) for img in self.filter(self.imgs)]
        self.masks = self.filter(self.masks)

        self.colormap2label = self.build_colormap2label()

    def get_voc_images(self, voc_dir, is_train):
        """Read all VOC feature and label images."""
        txt_fname = os.path.join(
            voc_dir, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt"
        )
        mode = torchvision.io.image.ImageReadMode.RGB
        with open(txt_fname, "r") as f:
            images = f.read().split()
        features, labels = [], []
        for _, fname in enumerate(images):
            features.append(
                torchvision.io.read_image(os.path.join(voc_dir, "JPEGImages", f"{fname}.jpg"))
            )
            labels.append(
                torchvision.io.read_image(
                    os.path.join(voc_dir, "SegmentationClass", f"{fname}.png"), mode
                )
            )
        return (features, labels)

    def voc_rand_crop(self, img, mask, height, width):
        """Randomly crop for both feature and label images."""
        rect = transforms.RandomCrop.get_params(img, (height, width))
        img = transforms.functional.crop(img, *rect)
        mask = transforms.functional.crop(mask, *rect)
        return img, mask

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [
            img
            for img in imgs
            if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])
        ]

    def build_colormap2label(self):
        """Build an RGB color to label mapping for segmentation."""
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(self.voc_colormap):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def voc_label_indices(self, colormap, colormap2label):
        """Map an RGB color to a label."""
        colormap = colormap.permute(1, 2, 0).numpy().astype("int32")
        idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
        return colormap2label[idx]

    def __getitem__(self, index):
        img, mask = self.voc_rand_crop(self.imgs[index], self.masks[index], *self.crop_size)
        mask = self.voc_label_indices(mask, self.build_colormap2label())
        return img, mask

    def __len__(self):
        return len(self.imgs)

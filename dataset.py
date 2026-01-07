import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset
TRAIN_PATH = "/data/ykx/MSRS/train"
VAL_PATH = "/data/ykx/MSRS/test"


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=A.Compose([ToTensorV2()]), mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == "train":
            self.files1 = natsorted(glob.glob(TRAIN_PATH + "/ir/*"))
            self.files2 = natsorted(glob.glob(TRAIN_PATH + "/vi/*"))
        else:
            self.files1 = natsorted(glob.glob(VAL_PATH + "/ir/*"))
            self.files2 = natsorted(glob.glob(VAL_PATH + "/vi/*"))

    def __getitem__(self, index):
        try:
            image_ir = Image.open(self.files1[index]).convert('L')
            image_vi = Image.open(self.files2[index]).convert('RGB')
            image_ir = np.array(image_ir)
            image_vi = np.array(image_vi)
            
            # ToTensorV2 不会自动归一化，需要先转换为 float32 并除以 255
            image_ir = image_ir.astype(np.float32) / 255.0
            image_vi = image_vi.astype(np.float32) / 255.0
            
            augmented = self.transform(image=image_vi, image_ir=image_ir)
            vi_tensor = augmented["image"]       # (3, H, W), range [0, 1]
            ir_tensor = augmented["image_ir"]    # (1, H, W), range [0, 1]
            return ir_tensor, vi_tensor

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return min(len(self.files1), len(self.files2))


# 训练增强：两张图共享同一组随机变换
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(height=256, width=256),
        ToTensorV2(),  # 同时作用到 image 和 additional_targets
    ],
    additional_targets={
        "image_ir": "image",
    },
)

val_transform = A.Compose(
    [
        ToTensorV2(),
    ],
    additional_targets={
        "image_ir": "image",
    },
)


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=train_transform, mode="train"),
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=val_transform, mode="val"),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=False,
)


if __name__ == "__main__":
    for ir, vi in trainloader:
        print(ir.shape, vi.shape)
        break
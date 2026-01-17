import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from natsort import natsorted
import numpy as np
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_PATH = "/data/ykx/MSRS/train"
TEST_PATH = "/data/ykx/MSRS/test"

VAL_RATIO = 0.03  
RANDOM_SEED = 42  
NUM_WORKERS = 4
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=A.Compose([ToTensorV2()]), 
                 files1=None, files2=None, data_path=None):
        """
        Args:
            transforms_: 数据增强
            files1: IR 文件列表（如果提供，直接使用）
            files2: VI 文件列表（如果提供，直接使用）
            data_path: 数据路径（如果 files1/files2 未提供，则从该路径读取）
        """
        self.transform = transforms_
        
        if files1 is not None and files2 is not None:
            self.files1 = files1
            self.files2 = files2
        elif data_path is not None:
            # 从指定路径读取
            self.files1 = natsorted(glob.glob(data_path + "/ir/*"))
            self.files2 = natsorted(glob.glob(data_path + "/vi/*"))
        else:
            # 默认使用训练集路径
            self.files1 = natsorted(glob.glob(TRAIN_PATH + "/ir/*"))
            self.files2 = natsorted(glob.glob(TRAIN_PATH + "/vi/*"))

        self.length = min(len(self.files1), len(self.files2))

    def __getitem__(self, index):

        index = index % self.length
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


    def __len__(self):
        return min(len(self.files1), len(self.files2))


# 训练增强：两张图共享同一组随机变换
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(height=480, width=480),
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


def get_train_val_datasets():
    """
    从训练集中划分训练集和验证集
    
    Returns:
        train_dataset: 训练集 Dataset
        val_dataset: 验证集 Dataset
    """
    # 获取所有文件路径
    all_files1 = natsorted(glob.glob(TRAIN_PATH + "/ir/*"))
    all_files2 = natsorted(glob.glob(TRAIN_PATH + "/vi/*"))
    
    # 确保两个列表长度一致
    min_len = min(len(all_files1), len(all_files2))
    all_files1 = all_files1[:min_len]
    all_files2 = all_files2[:min_len]
    
    # 随机打乱（使用固定种子保证可复现）
    indices = list(range(min_len))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)
    
    # 划分训练集和验证集
    val_size = int(min_len * VAL_RATIO)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 创建训练集和验证集
    train_files1 = [all_files1[i] for i in train_indices]
    train_files2 = [all_files2[i] for i in train_indices]
    val_files1 = [all_files1[i] for i in val_indices]
    val_files2 = [all_files2[i] for i in val_indices]
    
    train_dataset = Hinet_Dataset(
        transforms_=train_transform,
        files1=train_files1,
        files2=train_files2,
    )
    val_dataset = Hinet_Dataset(
        transforms_=val_transform,
        files1=val_files1,
        files2=val_files2
    )
    
    return train_dataset, val_dataset


# 训练模式：从训练集中划分训练集和验证集
train_dataset, val_dataset = get_train_val_datasets()

g = torch.Generator()
g.manual_seed(RANDOM_SEED)

trainloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
    drop_last=True,
)

valloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
    drop_last=False,
)

# 测试模式：使用测试集
testloader = DataLoader(
    Hinet_Dataset(transforms_=val_transform, data_path=TEST_PATH),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
    drop_last=False,
)


if __name__ == "__main__":
    # 显示数据集信息
    print("=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"验证集比例: {VAL_RATIO * 100:.1f}%")
    print(f"随机种子: {RANDOM_SEED}")
    print()
    test_dataset = Hinet_Dataset(transforms_=val_transform, data_path=TEST_PATH)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print()

    
    # 测试数据加载
    print("测试数据加载...")
    for ir, vi in trainloader:
        print(f"训练集 batch - IR shape: {ir.shape}, VI shape: {vi.shape}")
        break
    
    for ir, vi in valloader:
        print(f"验证集 batch - IR shape: {ir.shape}, VI shape: {vi.shape}")
        break
    
    for ir, vi in testloader:
        print(f"测试集 batch - IR shape: {ir.shape}, VI shape: {vi.shape}")
        break
    print("=" * 60)
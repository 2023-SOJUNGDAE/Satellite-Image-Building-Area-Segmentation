import cv2
import pandas as pd
import numpy as np
import os
import sys
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from backbones_unet.model.unet import Unet

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _format_logs(logs):
    str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
    s = ", ".join(str_logs)
    return s

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class SatelliteDatasetRandomCrop(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, data_set='train', n_crops=20):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.n_crops = n_crops
        self.crop_size = (224, 224)
        self.data_set = data_set

    def random_crop(self, image, mask):
        image_patches, mask_patches = [], []

        for _ in range(self.n_crops):
            crop = A.RandomCrop(height=self.crop_size[0], width=self.crop_size[1])
            augmented = crop(image=image, mask=mask)
            image_patches.append(augmented['image'])
            mask_patches.append(augmented['mask'])

        return image_patches, mask_patches

    def sliding_window(self, image, stepSize, windowSize, overlap=24):
        patch_count = 0
        y_start, y_end = 0, windowSize[1]
        x_start, x_end = 0, windowSize[0]
        for _ in range(5):
            for _ in range(5):
                yield (x_start, y_start, image[y_start:y_end, x_start:x_end])
                x_start += stepSize - overlap
                x_end += stepSize - overlap
                patch_count += 1
            y_start += stepSize - overlap
            y_end += stepSize - overlap
            x_start, x_end = 0, windowSize[0]
        assert patch_count == 25, f"Patch count should be 25, but got {patch_count} instead"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))


        if self.data_set == 'train':
            patches_image, patches_mask = self.random_crop(image, mask)
            for i in range(self.n_crops):
                if self.transform:
                    augmented = self.transform(image=patches_image[i], mask=patches_mask[i])
                    patches_image[i] = augmented['image']
                    patches_mask[i] = augmented['mask']
        elif self.data_set == 'valid':
            patches_image = []
            patches_mask = []
            for (x, y, window_image) in self.sliding_window(image, stepSize=200, windowSize=(224, 224)):
                patches_image.append(window_image)

            for (x, y, window_mask) in self.sliding_window(mask, stepSize=200, windowSize=(224, 224)):
                patches_mask.append(window_mask)

            for i in range(len(patches_image)):
                if self.transform:
                    augmented = self.transform(image=patches_image[i], mask=patches_mask[i])
                    patches_image[i] = augmented['image']
                    patches_mask[i] = augmented['mask']

        patches_image = np.stack(patches_image, axis=0)  # Stacking the patches
        patches_mask = np.stack(patches_mask, axis=0)  # Stacking the patches
        
        return patches_image, patches_mask


# traindata 로딩에 사용
class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def sliding_window(self, image, stepSize, windowSize, overlap=24):
        patch_count = 0
        y_start, y_end = 0, windowSize[1]
        x_start, x_end = 0, windowSize[0]
        for _ in range(5):
            for _ in range(5):
                yield (x_start, y_start, image[y_start:y_end, x_start:x_end])
                x_start += stepSize - overlap
                x_end += stepSize - overlap
                patch_count += 1
            y_start += stepSize - overlap
            y_end += stepSize - overlap
            x_start, x_end = 0, windowSize[0]
        assert patch_count == 25, f"Patch count should be 25, but got {patch_count} instead"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        patches_image = []
        patches_mask = []

        for (x, y, window_image) in self.sliding_window(image, stepSize=200, windowSize=(224, 224)):
            patches_image.append(window_image)

        for (x, y, window_mask) in self.sliding_window(mask, stepSize=200, windowSize=(224, 224)):
            patches_mask.append(window_mask)

        for i in range(len(patches_image)):
            if self.transform:
                augmented = self.transform(image=patches_image[i], mask=patches_mask[i])
                patches_image[i] = augmented['image']
                patches_mask[i] = augmented['mask']

        patches_image = np.stack(patches_image, axis=0)  # Stacking the patches
        patches_mask = np.stack(patches_mask, axis=0)  # Stacking the patches
        return patches_image, patches_mask


class dice_loss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(dice_loss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)

        # Flatten the prediction and target arrays
        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()
        dice_score = (2. * intersection + self.smooth) / (prediction.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss


# Create an instance of the IoU score class
class IoUScore(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoUScore, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):

        # Binarize the outputs and targets
        outputs = (outputs > 0.5).float()
        targets = (targets > 0.5).float()

        # Flatten the prediction and target tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum() - intersection

        iou_score = (intersection + self.smooth) / (union + self.smooth)

        return iou_score


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

# Define separate transforms for training and validation
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),

        A.RandomRotate90(p=0.3),
        A.Transpose(p=0.3),

        A.RandomBrightnessContrast(p=0.25), # 0.3 init
        # A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3), # 0.3 init

        A.GridDistortion(p=0.15), # 0.2 init
        A.OneOf([
          A.GaussianBlur(p=0.7),
          A.GaussNoise(p=0.3),
        ], p=0.1),

        A.Normalize(),
        ToTensorV2()
    ]
)

valid_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2()
    ]
)

# full_dataset을 분할

torch.manual_seed(1)
full_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)

train_size = int(0.8 * len(full_dataset))  # Use % of the data for training
valid_size = len(full_dataset) - train_size  # The rest for validation

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
train_dataset = copy.deepcopy(train_dataset)
valid_dataset = copy.deepcopy(valid_dataset)

# 각 데이터셋에 적절한 transform을 설정
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = valid_transform
valid_dataset.dataset.data_set = 'valid'

# Create dataloaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=28)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=16)

backbone = 'convnext_large_in22ft1k'
model = Unet(backbone=backbone, in_channels=3, num_classes=1).to(device=device)


criterion = dice_loss()
init_lr = 8e-6  # 초기 학습률 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-7)
max_epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=4, eta_min=5e-6,
)
    
iou_score_calculator = IoUScore().to(device)

save_path = '../saved_model'
log_path = '../logs'
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

best_iou_score = .8078
# model.load_state_dict(torch.load(f'{save_path}/{backbone}_best_model.pth'))

# training loop
for epoch in range(max_epochs):  # epoch
    print('\nEpoch: ', epoch)
    model.train()
    
    with tqdm(
        train_dataloader,
        desc='train',
        file=sys.stdout,
        disable=False,
    ) as iterator:
        train_logs = {}
        loss_meter = AverageValueMeter()
        iou_meter = AverageValueMeter()
        
        for images, masks in iterator:
            num_patches = images.size(1)
            for i in range(num_patches):
                image, mask = images[:, i].float().to(device), masks[:, i].float().to(device)
                optimizer.zero_grad()
                
                output = model(image)
                loss = criterion(output, mask.unsqueeze(1))
                
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'dice_loss': loss_meter.mean}
                train_logs.update(loss_logs)
                
                iou_score = iou_score_calculator(output, mask.unsqueeze(1)).cpu().detach().numpy()
                iou_meter.add(iou_score)
                iou_logs = {'iou_score': iou_meter.mean}
                train_logs.update(iou_logs)
                
                lr_logs = {'lr': scheduler.get_last_lr()[0]}
                train_logs.update(lr_logs)
                
                s = _format_logs(train_logs)
                iterator.set_postfix_str(s)
                
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), f'{save_path}/{backbone}_epoch_{epoch}.pth')
        scheduler.step()  # Update learning rate for the next epoch

    # Validation loop
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation
        with tqdm(
            valid_dataloader,
            desc='valid',
            file=sys.stdout,
            disable=False,
        ) as iterator:
            valid_logs = {}
            loss_meter = AverageValueMeter()
            iou_meter = AverageValueMeter()
        
            for val_images, val_masks in iterator:
                val_num_patches = val_images.size(1)
                for i in range(val_num_patches):
                    val_image, val_mask = val_images[:, i].float().to(device), val_masks[:, i].float().to(device)
                    val_output = model(val_image)

                    val_loss = criterion(val_output, val_mask.unsqueeze(1))
                    
                    loss_value = val_loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {'dice_loss': loss_meter.mean}
                    valid_logs.update(loss_logs)
                    
                    iou_score = iou_score_calculator(val_output, val_mask.unsqueeze(1)).cpu().detach().numpy()
                    iou_meter.add(iou_score)
                    iou_logs = {'iou_score': iou_meter.mean}
                    valid_logs.update(iou_logs)
                    
                    s = _format_logs(valid_logs)
                    iterator.set_postfix_str(s)

    ep_logs = {'epoch': epoch}
    train_logs.update(ep_logs)
    valid_logs.update(ep_logs)
    
    train_log_entry = json.dumps(train_logs)
    valid_log_entry = json.dumps(valid_logs)

    # 로그 파일에 쓰기
    with open(f"{log_path}/{backbone}-logfile.txt", "a") as file:
        file.write(train_log_entry + "\n")
        file.write(valid_log_entry + "\n")
    
    if best_iou_score < valid_logs['iou_score']:
        best_iou_score = valid_logs['iou_score']
        torch.save(model.state_dict(), f'{save_path}/{backbone}_best_model.pth')
        print('Best model saved!')
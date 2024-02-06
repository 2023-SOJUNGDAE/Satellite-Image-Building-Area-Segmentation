from backbones_unet.model.unet import Unet
import cv2
import pandas as pd
import numpy as np
import timm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RandomCropDataLoader
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
    
# Define separate transforms for training and validation
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),

        A.RandomRotate90(p=0.3),
        A.Transpose(p=0.3),

        A.RandomBrightnessContrast(p=0.25), # 0.3 init
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3), # 0.3 init

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

import copy
from torch.utils.data import random_split

# full_dataset을 분할

torch.manual_seed(6)

full_dataset = SatelliteDataset(csv_file='train.csv', transform=train_transform)

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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-xlarge", num_labels=1, ignore_mismatched_sizes=True).to(device)


criterion = dice_loss()
init_lr = 8e-6  # 초기 학습률 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-8)
max_epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=8, T_mult=2, eta_min=1e-6,
)


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
    
iou_score_calculator = IoUScore().to(device)

best_val_iou = 0.0

# training loop
for epoch in range(max_epochs):  # epoch
    model.train()
    epoch_loss = 0
    epoch_iou_score = 0
    num_batches = 0
    for images, masks in tqdm(train_dataloader):
        num_patches = images.size(1)

        batch_loss = 0
        batch_iou_score = 0
        for i in range(num_patches):
            image = images[:, i].float().to(device)

            output = model(image)

            mask = masks[:, i].float().to(device)
            # print(mask.unsqueeze(1).shape)
            optimizer.zero_grad()
            # print(image.shape)
            # print(output.shape)
            loss = criterion(output.logits, mask.unsqueeze(1))
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_iou_score += iou_score_calculator(output.logits, mask.unsqueeze(1)).item()
            # tqdm.write(f'loss_item: {loss.item()}')  # Use tqdm.write instead of print
            # sys.stdout.flush()  # Ensure the output is immediately displayed

        epoch_loss += batch_loss / num_patches
        epoch_iou_score += batch_iou_score / num_patches
        num_batches += 1

    torch.save(model.state_dict(), f'convnext_uper_xl_weight/best.pth')
    scheduler.step()  # Update learning rate for the next epoch
    print(f'Epoch {epoch}, Loss: {epoch_loss/num_batches}, IoU: {epoch_iou_score/num_batches}, Learning rate: {scheduler.get_last_lr()[0]}')

    # Validation loop
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation
        val_loss = 0
        val_iou_score = 0  # Initialize total IoU score for this epoch
        val_num_batches = 0
        for val_images, val_masks in tqdm(valid_dataloader):
            val_num_patches = val_images.size(1)

            val_batch_loss = 0
            val_batch_iou_score = 0
            for i in range(val_num_patches):
                val_image = val_images[:, i].float().to(device)
                val_mask = val_masks[:, i].float().to(device)

                val_output = model(val_image)


                val_loss_item = criterion(val_output.logits, val_mask.unsqueeze(1))

                val_batch_loss += val_loss_item.item()
                val_batch_iou_score += iou_score_calculator(val_output.logits, val_mask.unsqueeze(1)).item()  # Calculate IoU score for this batch

            val_loss += val_batch_loss / val_num_patches
            val_iou_score += val_batch_iou_score / val_num_patches

            val_num_batches += 1

        if val_iou_score/val_num_batches > best_val_iou:
            best_val_iou = val_iou_score/val_num_batches
            torch.save(model.state_dict(), f'convnext_uper_xl_weight_best.pth')
        
        print(f'Epoch {epoch}, Validation Loss: {val_loss/val_num_batches}, Validation IoU: {val_iou_score/val_num_batches}, Best val Iou: {best_val_iou}')

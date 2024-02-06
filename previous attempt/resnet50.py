#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
# 
# torch.cuda.current_device()
print(torch.__version__)
torch.cuda.is_available()
torch.cuda.set_device(2)


# In[4]:


import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()


# In[5]:


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


# In[6]:


# 기존 dataloader - testdata 로딩에 사용
class SatelliteDataset1(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# In[7]:


class SatelliteDatasetRandomCrop(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, data_set='train'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.n_crops = 25
        self.crop_size = (224, 224)

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
# class SatelliteDataset(Dataset):
#     def __init__(self, csv_file, transform=None, infer=False):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
#         self.infer = infer

#     def sliding_window(self, image, stepSize, windowSize):
#         for y in range(0, image.shape[0], stepSize):
#             for x in range(0, image.shape[1], stepSize):
#                 if x + windowSize[0] > image.shape[1] or y + windowSize[1] > image.shape[0]:
#                     # If remaining pixels are less than window size, start patch from the end of the image
#                     yield (x, y, image[image.shape[0] - windowSize[1]:, image.shape[1] - windowSize[0]:])
#                 else:
#                     yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data.iloc[idx, 1]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_rle = self.data.iloc[idx, 2]
#         mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

#         patches_image = []
#         patches_mask = []

#         for (x, y, window_image) in self.sliding_window(image, stepSize=224, windowSize=(224, 224)):
#             patches_image.append(window_image)

#         for (x, y, window_mask) in self.sliding_window(mask, stepSize=224, windowSize=(224, 224)):
#             patches_mask.append(window_mask)

#         for i in range(len(patches_image)):
#             if self.transform:
#                 augmented = self.transform(image=patches_image[i], mask=patches_mask[i])
#                 patches_image[i] = augmented['image']
#                 patches_mask[i] = augmented['mask']

#         patches_image = np.stack(patches_image, axis=0)  # Stacking the patches
#         patches_mask = np.stack(patches_mask, axis=0)  # Stacking the patches
#         return patches_image, patches_mask





# dataset = SatelliteDatasetRandomCrop(csv_file='./train.csv', transform=transform)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=32)
# 1 in dataset[0][1].tolist()


# In[9]:


# UNetWithResnet50Encoder
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UNetWithResnet50Encoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


# In[10]:


import torch
import torch.nn.functional as F

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

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        # BCE Loss
        bce_loss = self.bce_with_logits(prediction, target)

        # Dice Loss
        prediction = torch.sigmoid(prediction)

        # Flatten the prediction and target arrays
        prediction_ = prediction.view(-1)
        target_ = target.view(-1)

        intersection = (prediction_ * target_).sum()
        dice_score = (2. * intersection + 1e-7) / (prediction_.sum() + target_.sum() + 1e-7)
        dice_loss = 1 - dice_score

        # Edge Loss
        edge_prediction = F.conv2d(prediction, torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]).to(prediction.device))
        edge_target = F.conv2d(target, torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]).to(target.device))
        edge_loss = self.bce_with_logits(edge_prediction, edge_target)
        
        # Combine losses
        # loss = bce_loss + dice_loss + edge_loss
        loss = ( dice_loss * 9 + edge_loss ) / 10
        # print(f'loss: {loss}')
        return loss
# In[ ]:


# # 기본 training

# # model 초기화

# model = UNetWithResnet50Encoder(n_class=1).to(device)

# # loss function과 optimizer 정의
# # criterion = torch.nn.BCEWithLogitsLoss()
# criterion = dice_loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# # training loop
# for epoch in range(50):  #50 에폭 동안 학습합니다.
#     model.train()
#     epoch_loss = 0
#     for images, masks in tqdm(dataloader):
#         images = images.float().to(device)
#         masks = masks.float().to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks.unsqueeze(1))
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

# In[8]:

from torch.utils.data import random_split

# Define separate transforms for training and validation
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4, p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.05, rotate_limit=30, p=0.5), 
        # A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
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

# Create the full dataset
full_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)

# Split into train and validation datasets
train_size = int(0.95 * len(full_dataset))  # Use % of the data for training
valid_size = len(full_dataset) - train_size  # The rest for validation

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# Update the transform of the validation dataset
valid_dataset.dataset.transform = valid_transform
valid_dataset.dataset.data_set = 'valid'
# Create dataloaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=32)
# In[11]:


# UNetWithResnet50Encoder 패치 단위 training
model = UNetWithResnet50Encoder(n_class=1).to(device)
weights = torch.load('./0711_model_epoch_77.pth')
model.load_state_dict(weights)

# device1 = torch.device("cuda:2")
# device2 = torch.device("cuda:3")

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs")
#     model = nn.DataParallel(model, device_ids=[device1, device2])

# model.to(device1)

# loss function과 optimizer 정의

criterion = dice_loss()
init_lr = 0.0001  # 초기 학습률 설정
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
max_epochs = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=2e-5,
)


# Create an instance of the IoU score class
class IoUScore(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoUScore, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)

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
# Validation loop
# for epoch in range(max_epochs):  # epoch
#     model.eval()  # set the model to evaluation mode
#     with torch.no_grad():  # Turn off gradients for validation
#         val_loss = 0
#         val_iou_score = 0  # Initialize total IoU score for this epoch
#         val_num_batches = 0
#         for val_images, val_masks in tqdm(valid_dataloader):
#             val_num_patches = val_images.size(1)

#             val_batch_loss = 0
#             val_batch_iou_score = 0
#             for i in range(val_num_patches):
#                 val_image = val_images[:, i].float().to(device)
#                 val_mask = val_masks[:, i].float().to(device)
#                 val_output = model(val_image)

#                 val_loss_item = criterion(val_output, val_mask.unsqueeze(1))
#                 val_batch_loss += val_loss_item.item()
#                 val_batch_iou_score += iou_score_calculator(val_output, val_mask.unsqueeze(1)).item()  # Calculate IoU score for this batch

#             val_loss += val_batch_loss / val_num_patches
#             val_iou_score += val_batch_iou_score / val_num_patches

#             val_num_batches += 1

#         print(f'Epoch {epoch+1}, Validation Loss: {val_loss/val_num_batches}, Validation IoU: {val_iou_score/val_num_batches}')


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
            mask = masks[:, i].float().to(device)
            optimizer.zero_grad()
            output = model(image)

            loss = criterion(output, mask.unsqueeze(1))
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_iou_score += iou_score_calculator(output, mask.unsqueeze(1)).item()
            # tqdm.write(f'loss_item: {loss.item()}')  # Use tqdm.write instead of print
            # sys.stdout.flush()  # Ensure the output is immediately displayed

        epoch_loss += batch_loss / num_patches
        epoch_iou_score += batch_iou_score / num_patches
        num_batches += 1

    torch.save(model.state_dict(), f'0711_model_epoch_{epoch+77}.pth')
    scheduler.step()  # Update learning rate for the next epoch
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/num_batches}, IoU: {epoch_iou_score/num_batches}, Learning rate: {scheduler.get_last_lr()[0]}')

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

                val_loss_item = criterion(val_output, val_mask.unsqueeze(1))

                val_batch_loss += val_loss_item.item()
                val_batch_iou_score += iou_score_calculator(val_output, val_mask.unsqueeze(1)).item()  # Calculate IoU score for this batch

            val_loss += val_batch_loss / val_num_patches
            val_iou_score += val_batch_iou_score / val_num_patches

            val_num_batches += 1

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss/val_num_batches}, Validation IoU: {val_iou_score/val_num_batches}')




# In[ ]:


transform1 = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)
test_dataset = SatelliteDataset1(csv_file='./test.csv', transform=transform1, infer=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)
# test_dataset[0]



# In[ ]:


# # 가중치를 저장
# save_path = './Unet_Resnet50_dynlr0.0001_weights.pth'
# torch.save(model.state_dict(), save_path)


# In[ ]:


# # (저장된 모델이 있다면) 모델 가중치 불러오기
# model = UNetWithResnet50Encoder(n_class=1).to(device)
# weights = torch.load('./model_epoch_29.pth')
# model.load_state_dict(weights)


# In[ ]:


with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.5).astype(np.uint8) # Threshold

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)


# In[ ]:


submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result


# In[ ]:


submit.to_csv('./submit.csv', index=False)


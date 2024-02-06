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
torch.cuda.set_device(2)

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

# RandomCropDataLoader

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

# import torchvision
# from torchvision.models import resnet50, ResNet50_Weights

# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )

# class UNetWithResnet50Encoder(nn.Module):
#     def __init__(self, n_class=1):
#         super().__init__()

#         self.base_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         self.base_layers = list(self.base_model.children())

#         self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
#         self.layer1_1x1 = convrelu(256, 256, 1, 0)
#         self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
#         self.layer2_1x1 = convrelu(512, 512, 1, 0)
#         self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
#         self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
#         self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
#         self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
#         self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
#         self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
#         self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

#         self.conv_last = nn.Conv2d(64, n_class, 1)

#     def forward(self, input):
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)

#         layer0 = self.layer0(input)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)

#         layer4 = self.layer4_1x1(layer4)
#         x = self.upsample(layer4)
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)

#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)

#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)

#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)

#         x = self.upsample(x)
#         x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)

#         out = self.conv_last(x)

#         return out


# loss function
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


# Define separate transforms for training and validation
train_transform = A.Compose(
    [
        # A.HorizontalFlip(p=0.4),
        # A.VerticalFlip(p=0.4),
        # A.Rotate(limit=90, p=0.4),
        # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=0.4),
        # A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4, p=0.4),
        # A.RandomScale(scale_limit=(0, 0.2), p=0.3),
        # A.Resize(224,224),
        # A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        # A.GaussNoise(p=0.2),
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

from torch.utils.data import random_split

# Create the full dataset
full_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)
# train_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)

# Split into train and validation datasets
train_size = int(0.8 * len(full_dataset))  # Use % of the data for training
valid_size = len(full_dataset) - train_size  # The rest for validation

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# # Update the transform of the validation dataset
# valid_dataset.dataset.transform = valid_transform
# valid_dataset.dataset.data_set = 'valid'

# Create dataloaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=32)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=32)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming'):
    net.apply(weights_init_kaiming)

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class UNet_3Plus(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        filters = [64, 256, 512, 1024, 2048] # [64, 128, 256, 512, 1024] [64, 256, 512, 1024, 2048]
        self.ResNet101 = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2) #
        self.ResNet101.conv1.stride = (1, 1)

        # self.ResNet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) #
        # self.ResNet50.conv1.stride = (1, 1)

        ## -------------Encoder--------------
        # self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        
        self.conv1 = nn.Sequential(
                            self.ResNet101.conv1, # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                            self.ResNet101.bn1,   # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            self.ResNet101.relu   # (relu): ReLU(inplace=True)
                        )
                        
        self.init_pool = self.ResNet101.maxpool # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = self.ResNet101.layer1
        self.conv3 = self.ResNet101.layer2
        self.conv4 = self.ResNet101.layer3
        self.conv5 = self.ResNet101.layer4

        # self.conv1 = nn.Sequential(
        #                     self.ResNet50.conv1, # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #                     self.ResNet50.bn1,   # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #                     self.ResNet50.relu   # (relu): ReLU(inplace=True)
        #                 )
                        
        # self.init_pool = self.ResNet50.maxpool # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # self.conv2 = self.ResNet50.layer1
        # self.conv3 = self.ResNet50.layer2
        # self.conv4 = self.ResNet50.layer3
        # self.conv5 = self.ResNet50.layer4

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        # h1 = self.conv1(inputs)  # h1->320*320*64

        # h2 = self.maxpool1(h1)
        # h2 = self.conv2(h2)  # h2->160*160*128

        # h3 = self.maxpool2(h2)
        # h3 = self.conv3(h3)  # h3->80*80*256

        # h4 = self.maxpool3(h3)
        # h4 = self.conv4(h4)  # h4->40*40*512

        # h5 = self.maxpool4(h4)
        # hd5 = self.conv5(h5)  # h5->20*20*1024

        h1 = self.conv1(inputs)

        p1 = self.init_pool(h1)
        h2 = self.conv2(p1)

        h3 = self.conv3(h2)
        
        h4 = self.conv4(h3)
        
        hd5 = self.conv5(h4)
        
        # print(h1.shape)
        # print(h2.shape)
        # print(h3.shape)
        # print(h4.shape)
        # print(hd5.shape)
        # hd5 = self.upsample(self.conv5(h4))

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels


        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1

# UNetWithResnet50Encoder 패치 단위 training
# model = UNetWithResnet50Encoder(n_class=1).to(device)
# model = UNet_3Plus().to(device)
# model = UNet_3Plus()
# if torch.cuda.is_available():
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # 기본적으로 1번 GPU를 사용
#     model = model.to(device) # 모델을 GPU로 옮깁니다.
    
#     # 가중치를 로드합니다.
#     weights = torch.load('./0712_Unet3_Resnet101_epoch15.pth')
#     model.load_state_dict(weights)
#     model = nn.DataParallel(model, device_ids=[1, 2]) # 여기서 device_ids를 이용해 사용할 GPU를 명시합니다.

# 멀티GPU-> 단일GPU
model = UNet_3Plus()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 기본적으로 1번 GPU를 사용

# 모델을 GPU로 옮깁니다.
model = model.to(device)

# 가중치를 로드합니다.
weights = torch.load('./0712_Unet3_Resnet101_epoch15.pth', map_location=device)

# DataParallel로 감싸져 있는 경우를 대비해 이를 제거합니다.
if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.load_state_dict(weights)


# weights = torch.load('./0712_Unet3_Resnet101_epoch15.pth', map_location=device)
# model.load_state_dict(weights)

criterion = dice_loss()
init_lr = 0.0002  # 초기 학습률 설정
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
max_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

iou_score_calculator = IoUScore().to(device)

# from torch.utils.bottleneck import summary
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
         # 프로파일링 시작
        # if epoch == 0 and num_batches == 7:
        #     with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #         for i in range(num_patches):
        #             image = images[:, i].float().to(device)
        #             mask = masks[:, i].float().to(device)
        #             optimizer.zero_grad()
        #             output = model(image)
        #             loss = criterion(output, mask.unsqueeze(1))
        #             loss.backward()
        #             optimizer.step()

        #             batch_loss += loss.item()
        #             batch_iou_score += iou_score_calculator(output, mask.unsqueeze(1)).item()
        #     print(prof.key_averages().table(sort_by="cuda_time_total"))
        # # 프로파일링 종료
        
        # else:
        for i in range(num_patches):
            image = images[:, i].float().to(device)
            mask = masks[:, i].float().to(device)
            optimizer.zero_grad()
            output = model(image)
            # print(f'{output.shape}, {mask.unsqueeze(1).shape}')
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

    torch.save(model.state_dict(), f'0712_Unet3_Resnet101_epoch{(epoch + 1) + 15}.pth')
    scheduler.step()  # Update learning rate for the next epoch
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/num_batches}, IoU: {epoch_iou_score/num_batches}, Learning rate: {scheduler.get_last_lr()[0]}')

    # # Validation loop
    # model.eval()  # set the model to evaluation mode
    # with torch.no_grad():  # Turn off gradients for validation
    #     val_loss = 0
    #     val_iou_score = 0  # Initialize total IoU score for this epoch
    #     val_num_batches = 0
    #     for val_images, val_masks in tqdm(valid_dataloader):
    #         val_num_patches = val_images.size(1)

    #         val_batch_loss = 0
    #         val_batch_iou_score = 0
    #         for i in range(val_num_patches):
    #             val_image = val_images[:, i].float().to(device)
    #             val_mask = val_masks[:, i].float().to(device)
    #             val_output = model(val_image)

    #             val_loss_item = criterion(val_output, val_mask.unsqueeze(1))

    #             val_batch_loss += val_loss_item.item()
    #             val_batch_iou_score += iou_score_calculator(val_output, val_mask.unsqueeze(1)).item()  # Calculate IoU score for this batch

    #         val_loss += val_batch_loss / val_num_patches
    #         val_iou_score += val_batch_iou_score / val_num_patches

    #         val_num_batches += 1

    #     print(f'Epoch {epoch+1}, Validation Loss: {val_loss/val_num_batches}, Validation IoU: {val_iou_score/val_num_batches}')

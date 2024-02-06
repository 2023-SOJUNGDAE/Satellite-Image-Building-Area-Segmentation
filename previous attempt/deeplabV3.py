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
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        A.Rotate(limit=90, p=0.4),
        # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=0.5),

        # A.RandomScale(scale_limit=0.3, p=0.3),
        # A.PadIfNeeded(min_height=224, min_width=224),

        # color transforms
        # A.OneOf(
        #     [
        #         A.RandomBrightnessContrast(p=1),
        #         A.RandomGamma(p=1),
        #         A.ChannelShuffle(p=0.2),
        #         A.HueSaturationValue(p=1),
        #         A.RGBShift(p=1),
        #     ], p=0.5),

        # noise transforms
        # A.OneOf([
        #   A.GaussianBlur(p=1),
        #   A.GaussNoise(p=1),
        # ], p=0.1),

        # A.Resize(224, 224),
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
torch.manual_seed(44)
full_dataset = SatelliteDatasetRandomCrop(csv_file='./train.csv', transform=train_transform)

train_size = int(0.8 * len(full_dataset))  # Use % of the data for training
valid_size = len(full_dataset) - train_size  # The rest for validation

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# deepcopy를 이용하여 데이터셋을 복사
train_dataset = copy.deepcopy(train_dataset)
valid_dataset = copy.deepcopy(valid_dataset)

# 각 데이터셋에 적절한 transform을 설정
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = valid_transform
valid_dataset.dataset.data_set = 'valid'

# Create dataloaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=32)

# full_dataset = SatelliteDatasetRandomCrop(csv_file='./train.csv', transform=train_transform)

# # Split into train and validation datasets
# train_size = int(0.8 * len(full_dataset))  # Use % of the data for training
# valid_size = len(full_dataset) - train_size  # The rest for validation

# train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# # Update the transform of the validation dataset
# valid_dataset.dataset.transform = valid_transform
# valid_dataset.dataset.data_set = 'valid'

# # Create dataloaders for train and validation datasets
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
# valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=32)

# import pandas as pd
# from sklearn.model_selection import train_test_split

# # # Load the full dataset
# # full_data = pd.read_csv('./train.csv')
# # _, valid_data = train_test_split(full_data, test_size=0.25, random_state=42)
# # valid_data.to_csv('./valid_split.csv', index=False)

# valid_dataset = SatelliteDatasetRandomCrop(csv_file='./valid_split.csv', transform=valid_transform, n_crops=5)
# valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=32)

# # Create dataloaders for train and validation datasets
# train_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)
# # train_dataset = SatelliteDataset(csv_file='./train.csv', transform=train_transform)
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32)



# ASPP
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)

# decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)

# backbone
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm,
                 pretrained=False):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrain_dict = torch.load('./xception_pretrained.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                             yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                             yield p



# model = UNetWithResnet50Encoder(n_class=1).to(device)
# model = DeepLab(backbone='xception', num_classes=1).to(device)
# weights = torch.load('./0716_DeepLabV3+_model_epoch_28.pth', map_location=device)
# model.load_state_dict(weights)

# model = DeepLab(backbone='xception', num_classes=1)

# if torch.cuda.is_available():
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # 기본적으로 1번 GPU를 사용
#     model = model.to(device) # 모델을 GPU로 옮깁니다.
#     model = nn.DataParallel(model, device_ids=[1, 2]) # 여기서 device_ids를 이용해 사용할 GPU를 명시합니다.
#     # 가중치를 로드합니다.
#     weights = torch.load('./DeepLabV3Plus/0716_DeepLabV3+_model_epoch_95.pth')
#     model.load_state_dict(weights)
    
model = DeepLab(backbone='xception', num_classes=1).to(device)

if torch.cuda.is_available():
    device = torch.device('cuda:2') # 1번 GPU를 사용
    model = model.to(device) # 모델을 GPU로 옮깁니다.
    def strip_module_from_state_dict(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    weights = torch.load('./DeepLabV3Plus/0716_DeepLabV3+_model_epoch_95.pth', map_location=device)
    weights = strip_module_from_state_dict(weights)

    model.load_state_dict(weights)


criterion = dice_loss()
init_lr = 0.0001  # 초기 학습률 설정
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
max_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=2e-5,
)

iou_score_calculator = IoUScore().to(device)


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
            # print(mask.unsqueeze(1).shape)
            optimizer.zero_grad()
            output = model(image)
            # print(image.shape)
            # print(output.shape)
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

    torch.save(model.state_dict(), f'./DeepLabV3Plus/0716_DeepLabV3+_model_epoch_{epoch+1+95}.pth')
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

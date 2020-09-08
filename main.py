# -*- coding: utf-8 -*-

import numpy as np
import time
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu6(x + 3) / 6



class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.hsigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.hsigmoid(x)


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x



class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2

        self.use_connect = (stride == 1) and (in_channels == out_channels)

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out



class MobileNetV3Small(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNetV3Small, self).__init__()
        
        self.n_class = n_class

        layers = [
            [16, 16, 3, 2, "RE", True, 16],
            [16, 24, 3, 2, "RE", False, 72],
            [24, 24, 3, 1, "RE", False, 88],
            [24, 40, 5, 2, "RE", True, 96],
            [40, 40, 5, 1, "RE", True, 240],
            [40, 40, 5, 1, "RE", True, 240],
            [40, 48, 5, 1, "HS", True, 120],
            [48, 48, 5, 1, "HS", True, 144],
            [48, 96, 5, 2, "HS", True, 288],
            [96, 96, 5, 1, "HS", True, 576],
            [96, 96, 5, 1, "HS", True, 576],
        ]

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            h_swish(inplace=True),
        )

        self.block = []
        for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
            self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
        self.block = nn.Sequential(*self.block)


        self.out_conv = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1),
            SqueezeBlock(576),
            nn.BatchNorm2d(576),
            h_swish(inplace=True),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(576, 1024, kernel_size=1, stride=1),
            h_swish(inplace=True),
            nn.Conv2d(1024, self.n_class, kernel_size=1, stride=1),
        )


    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv(out)
        out = out.view(out.shape[0], -1)
        return out



def make_predictions(model, test_loader, device):

    model.eval()
    predictions = []

    with torch.set_grad_enabled(False):
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.data.cpu().numpy().argmax(1)
            predictions += list(preds)

    return predictions



def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
        ])
    
    # TEST DATA FROM IMAGE FOLDER    
    test_folder_path = './test_folder'
    test_set = torchvision.datasets.ImageFolder(test_folder_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    # MODEL LOADING
    MODEL_SAVE_PATH = './mobilenetv3small.pt'
    model = MobileNetV3Small(n_class=100).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # MAKE PREDICTIONS
    start = time.time()
    predictions = make_predictions(model, test_loader, device)
    end = time.time()
    
    print("Predictions:", predictions)
    print("Process time per image: {:0.4f} ms".format((end - start)/len(test_set)*1e3) )



if __name__ == "__main__":
    main()
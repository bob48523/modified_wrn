import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class trunk(nn.Module):
    def __init__(self, in_planes, planes):
        super(trunk, self).__init__()
        self.res1 = PreActBottleneck(in_planes, planes)
        self.res2 = PreActBottleneck(planes*4, planes)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        return out

class mask_1(nn.Module):
    def __init__(self, in_planes, planes):
        super(mask_1, self).__init__()
        #downsample_layer = []
        #upsample_layer = []
        #skip_layer = []
        #for i in range(1, num_pool):
        #    layers.append(nn.MaxPool2d(3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = PreActBottleneck(in_planes, planes)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res2 = PreActBottleneck(planes*4, planes)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res3_1 = PreActBottleneck(planes*4, planes)
        self.res3_2 = PreActBottleneck(planes*4, planes)
        
        self.interp1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res4 = PreActBottleneck(planes*4, planes)
        self.interp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res5 = PreActBottleneck(planes*4, planes)
        self.interp3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res6_1 = PreActBottleneck(planes*4, planes)
        self.res6_2 = PreActBottleneck(planes*4, planes)
        
        self.skip_1 = PreActBottleneck(planes*4, planes)
        self.skip_2 = PreActBottleneck(planes*4, planes)
        
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv1 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.res1(self.pool1(x))
        out2 = self.res2(self.pool2(out1))
        out3 = self.pool3(out2)
        out4 = self.res3_2(self.res3_1(out3))
        
        out5 = self.interp1(out4)
        out6 = out5+self.skip_1(out2)
        out7 = self.interp2(self.res4(out6))
        out8 = out7+self.skip_2(out1)
        out9 = self.interp3(self.res5(out8))
        out10 = self.res6_2(self.res6_1(out9))
        out = self.conv1(F.relu(self.bn1(out10)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out10

class mask_2(nn.Module):
    def __init__(self, in_planes, planes):
        super(mask_2, self).__init__()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = PreActBottleneck(in_planes, planes)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res2_1 = PreActBottleneck(planes*4, planes)
        self.res2_2 = PreActBottleneck(planes*4, planes)
        
        self.interp1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res3 = PreActBottleneck(planes*4, planes)
        self.interp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res4_1 = PreActBottleneck(planes*4, planes)
        self.res4_2 = PreActBottleneck(planes*4, planes)
        
        self.skip_1 = PreActBottleneck(planes*4, planes)
        
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv1 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.res1(self.pool1(x))
        out2 = self.pool2(out1)
        out3 = self.res2_2(self.res2_1(out2))
        
        out4 = self.interp1(out3)
        out5 = out4+self.skip_1(out1)
        out6 = self.interp2(self.res3(out5))
        out7 = self.res4_2(self.res4_1(out6))

        out = self.conv1(F.relu(self.bn1(out7)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out

class mask_3(nn.Module):
    def __init__(self, in_planes, planes):
        super(mask_3, self).__init__()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1_1 = PreActBottleneck(in_planes, planes)
        self.res1_2 = PreActBottleneck(planes*4, planes)
        
        self.interp1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res2_1 = PreActBottleneck(planes*4, planes)
        self.res2_2 = PreActBottleneck(planes*4, planes)

        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv1 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.res1_2(self.res1_1(self.pool1(x)))
        out2 = self.res2_2(self.res2_1(self.interp1(out1)))
        out = self.conv1(F.relu(self.bn1(out2)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out

class attention(nn.Module):
    def __init__(self, in_planes, planes, num_pool):
        super(attention, self).__init__()
        self.res1 = PreActBottleneck(in_planes, planes)
        if num_pool == 3:
            self.mask = mask_1(planes*4, planes)
        elif num_pool == 2:
            self.mask = mask_2(planes*4, planes)
        else:
            self.mask = mask_3(planes*4, planes)
        self.trunk = trunk(planes*4, planes)
        self.res2 = PreActBottleneck(planes*4, planes)

    def forward(self, x):
        out1 = self.res1(x)
        out2 = self.trunk(out1)
        out3 = self.mask(out1)
        out = out3*out2
        out += out2
        out = self.res2(out)
        return out

class ResAttentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResAttentNet, self).__init__()
        in_planes = 64
        num_atten = 2
        planes = 16
        self.conv1 = nn.Conv2d(3,in_planes, kernel_size = 3, padding = 1)
        
        self.stage1 = self._make_stage(in_planes, planes, 3, num_atten)   
        self.res1 = PreActBottleneck(planes*4, planes*2, stride=2)
         
        in_planes = planes*8
        planes *=2 
        self.stage2 = self._make_stage(in_planes, planes, 2, num_atten)
        self.res2 = PreActBottleneck(planes*4, planes*2, stride=2)
       
        in_planes = planes*8
        planes *=2 
        self.stage3 = self._make_stage(in_planes, planes, 1, num_atten)

        self.res3 = PreActBottleneck(planes*4, planes*2, stride=2)
        in_planes = planes*8
        planes *=2 
        self.res4 = PreActBottleneck(in_planes, planes, stride=1)
        self.res5 = PreActBottleneck(planes*4, planes, stride=1)
        self.res6 = PreActBottleneck(planes*4, planes, stride=1)
        self.fc = nn.Linear(planes*4, num_classes)

    def _make_stage(self, in_planes, planes, num_pool, num_atten=1):
        atten_layers = []
        for i in range(num_atten):
            atten_layers.append(attention(in_planes, planes, num_pool))
            in_planes = planes*4
        return nn.Sequential(*atten_layers)
         
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.res1(out)
        out = self.stage2(out)
        out = self.res2(out)
        out = self.stage3(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




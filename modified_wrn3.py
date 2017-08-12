import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Inception(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super(Inception, self).__init__()
        # 1x1 conv branch
        n_outplane1 = out_planes // 3
        n_outplane2 = n_outplane1
        n_outplane3 = out_planes - n_outplane1*2;

        self.b11 = nn.Sequential(
            nn.Conv2d(in_planes, n_outplane1, stride=stride, kernel_size=1,bias=False),
            nn.BatchNorm2d(n_outplane1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b12 = nn.Sequential(
            nn.Conv2d(in_planes, n_outplane2, stride=stride, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_outplane2),
            nn.ReLU(True),
            nn.Conv2d(n_outplane2, n_outplane2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_outplane2),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b13 = nn.Sequential(
            nn.Conv2d(in_planes, n_outplane3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(n_outplane3),
            nn.ReLU(True),
            nn.Conv2d(n_outplane3, n_outplane3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_outplane3),
            nn.ReLU(True),
            nn.Conv2d(n_outplane3, n_outplane3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_outplane3),
            nn.ReLU(True),
        )

   
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        #self.dropout = nn.Dropout(p=dropout_rate)

        # 1x1 conv branch
        #self.b21 = nn.Sequential(
        #    nn.Conv2d(out_planes, n_outplane1, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(n_outplane1),
        #    nn.ReLU(True),
        #)

        # 1x1 conv -> 3x3 conv branch
        #self.b22 = nn.Sequential(
        #    nn.Conv2d(out_planes, n_outplane2, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(n_outplane2),
        #    nn.ReLU(True),
        #    nn.Conv2d(n_outplane2, n_outplane2, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(n_outplane2),
        #    nn.ReLU(True),
        #)

        # 1x1 conv -> 5x5 conv branch
        #self.b23 = nn.Sequential(
        #    nn.Conv2d(out_planes, n_outplane3, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(n_outplane3),
        #    nn.ReLU(True),
        #    nn.Conv2d(n_outplane3, n_outplane3, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(n_outplane3),
        #    nn.ReLU(True),
        #    nn.Conv2d(n_outplane3, n_outplane3, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(n_outplane3),
        #    nn.ReLU(True),
        #)
        #self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if in_planes != out_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, stride = stride, kernel_size = 1, bias=False)
            )
       

    def forward(self, x):
        y11 = self.b11(x)
        y12 = self.b12(x)
        y13 = self.b13(x)
        out1 = torch.cat([y11,y12,y13], 1)
        out1 = F.relu((self.conv1(out1)))
        
        #y21 = self.b21(out1)
        #y22 = self.b22(out1)
        #y23 = self.b23(out1)
        #out2 = torch.cat([y21,y22,y23], 1)
        #out2 = F.relu((self.conv1(out2)))
        
        out = out1 + self.shortcut(x)
        return out


class Cifar10_net(nn.Module):
    def __init__(self, block, num_blocks, dropout_rate, num_classes=10):
        super(Cifar10_net, self).__init__()
        self.in_planes = 64
        widen_factor = 4;
        width = [64*widen_factor,128*widen_factor,256*widen_factor]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.layer1 = self._make_layer(block, width[0], num_blocks[0], dropout_rate, 1)
        self.layer2 = self._make_layer(block, width[1], num_blocks[1], dropout_rate, 2)
        self.layer3 = self._make_layer(block, width[2], num_blocks[2], dropout_rate, 2)
        self.bn2 = nn.BatchNorm2d(width[2])
        self.fc = nn.Linear(width[2], num_classes)


    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride): 
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
 
        out = F.avg_pool2d(self.bn2(out), 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def cifar10Net26():
    return Cifar10_net(Inception, [3,4,6], 0.5, 10)


# test()

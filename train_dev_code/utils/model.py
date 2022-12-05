import torch.nn as nn
import torch.nn.functional as F

class nvidia_model(nn.Module):
    in_planes = [3, 24, 36, 48, 64, 64, 1152, 100, 50, 10, 2]
    kernel_size = [5, 5, 5, 3, 3]
    dropout_p = [0.45, 0.4, 0.4]
    def __init__(self):
        super(nvidia_model, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(self.in_planes[0], self.in_planes[1], self.kernel_size[0], stride=2)
        self.conv2 = nn.Conv2d(self.in_planes[1], self.in_planes[2], self.kernel_size[1], stride=2)
        self.conv3 = nn.Conv2d(self.in_planes[2], self.in_planes[3], self.kernel_size[2], stride=2)
        self.conv4 = nn.Conv2d(self.in_planes[3], self.in_planes[4], self.kernel_size[3])
        self.conv5 = nn.Conv2d(self.in_planes[4], self.in_planes[5], self.kernel_size[4])
        self.dropout1 = nn.Dropout(p=self.dropout_p[0])
        self.fc1 = nn.Linear(self.in_planes[6], self.in_planes[7])
        self.dropout2 = nn.Dropout(p=self.dropout_p[1])
        self.fc2 = nn.Linear(self.in_planes[7], self.in_planes[8])
        self.dropout3 = nn.Dropout(p=self.dropout_p[2])
        self.fc3 = nn.Linear(self.in_planes[8], self.in_planes[9])
        self.output = nn.Linear(self.in_planes[9], self.in_planes[10])

    # def print_layer(self, layer):
    #     print(layer.shape)
    #     print(type(layer))

    def forward(self, x):
        # self.print_layer(x)
        out = self.bn1(x)
        # self.print_layer(out)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.dropout1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = F.relu(self.fc2(out))
        out = self.dropout3(out)
        out = F.relu(self.fc3(out))
        out = self.output(out)
        return out

# '''
# Resnet.py
# '''
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                         kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         '''
#         If C7 flag is true, the BN layers are removed.
#         By default, the BN layers are attached.
#         '''
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def ResNet18(argflag):
#     global C7Flag
#     C7Flag = argflag
#     return ResNet(BasicBlock, [2, 2, 2, 2])
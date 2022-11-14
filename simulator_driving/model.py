import torch.nn as nn
import torch.nn.functional as F

class nvidia_model(nn.Module):
    in_planes = [3, 24, 36, 48, 64, 64, 1152, 100, 50, 10, 1]
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


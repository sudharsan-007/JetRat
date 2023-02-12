import torch.nn as nn
import torchvision

output_dim = 1 # Change to 1 if only predicting steering

class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.
        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)
        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=28224, out_features=100),
            # nn.Linear(in_features=48576, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=output_dim) 
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 224, 224)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


def ResNet18():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, output_dim)
    return model 

def ResNet34():
    # RESNET 34
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, output_dim)
    return model

def AlexNet():
    # ALEXNET
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, output_dim)
    return model

def VGG16():
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, output_dim)
    return model
    
## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        # output tensor: (8, 220, 220)
        # after one pool layer, this becomes (8, 110, 110)

        self.conv1 = nn.Conv2d(1, 8, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)

        # second conv layer: 8 inputs, 32 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (32, 108, 108)
        # after another pool layer this becomes (32, 54, 54);  no need to rounded down
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.dropout2 = nn.Dropout(p=0.3)

        # third conv layer: 64 inputs, 128 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output tensor will have dimensions: (32, 53, 53)
        # after another pool layer this becomes (64, 26, 26);  26.5 rounded down
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        # 128 outputs * the 2*2 filtered/pooled map size
        self.fc1 = nn.Linear(128 * 27 * 27, 2048)
        self.fc1_drop = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(2048, 512)
        self.fc2_drop = nn.Dropout(p=0.4)


        # finally, create 136 output channels (for the 136 classes)
        self.fc3 = nn.Linear(512, 136)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.xavier_uniform_(m.weight.data)
                I.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.Linear):
                I.kaiming_uniform_(m.weight.data)
                I.constant_(m.bias.data, 0.1)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout2(x)

        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout3(x)


        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x

## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    '''Define DCNN using NaimishNet architecture for facial key points detection
    
    '''

    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional 2D layer, kernel size: 4x4
        # input: (1, 224, 224)
        # output: (32, 221, 221) <- (224-4)/1 + 1
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # Max Pool layer, kernel size = 2, stride = 2
        # input: (32, 221, 221)
        # output: (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=.1)

        # Convolutional 2D layer, kernel size: 3x3 
        # input: (32, 110, 110)
        # output: (64, 108, 108) <- (110-3)/1 + 1
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Max Pool layer, kernel size = 2, stride = 2
        # input: (64, 108, 108)
        # output: (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=.2)
 
        # Convolutional 2D layer, kernel size: 2x2
        # input: (64, 54, 54)
        # output: (128, 53, 53) <- (54-2)/1 + 1
        self.conv3 = nn.Conv2d(64, 128, 2)

        # Max Pool layer, kernel size = 2, stride = 2
        # input: (128, 53, 53)
        # output: (128, 26, 26)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=.3)

        # Convolutional 2D layer, kernel size: 1x1
        # input: (128, 26, 26)
        # output: (256, 26, 26) <- (26-1)/1 + 1
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # Max Pool layer, kernel size = 2, stride = 2
        # input: (256, 26, 26)
        # output: (256, 13, 13)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=.4)    
        
        self.dense1 = nn.Linear(256*13*13, 7000)
        self.drop5 = nn.Dropout(p=.5)

        self.dense2 = nn.Linear(7000, 1000)
        self.drop6 = nn.Dropout(p=.6)

        self.output = nn.Linear(1000, 68*2)


        
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        
        # Flatten to 1D tensor 6400 <- 5*5*256
        # Flatten to 1D tensor 43264 <- 13*13*256
        x = x.view(x.size(0), -1)
        
        x = self.drop5(F.relu(self.dense1(x)))
        x = self.drop6(F.relu(self.dense2(x)))
        x = self.output(x)
        
        return x

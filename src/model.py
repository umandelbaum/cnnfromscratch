import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        #conv layer 1: gets 3x224x224, puts out 5 feature maps of 112*112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 5, kernel_size = 3, padding=1) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2) 
        
        #conv layer 2: gets 5x112x112, puts out 7 feature maps of 56*56
        self.conv2 = nn.Conv2d(in_channels=5, out_channels = 7, kernel_size = 3, padding=1) 
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2) 
        
        #conv layer 3: gets 7 56x56 feature maps, puts out 9 feature maps of 28*28
        self.conv3 = nn.Conv2d(in_channels=7, out_channels = 9, kernel_size = 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2) 
        
        #conv layer 4: gets 11 28x28 feature maps, puts out 11 feature maps of 14*14
        self.conv4 = nn.Conv2d(in_channels=9, out_channels = 11, kernel_size = 3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2) 
        
        #conv layer 5: gets 13 14x14 feature maps, puts out 13 feature maps of 7*7
        self.conv5 = nn.Conv2d(in_channels=11, out_channels = 13, kernel_size = 3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2,2) 
        
        #linear layer #1:  turns the 13 7x7 feature maps into a 100 values 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * 7 * 7, 100)
        self.dp1 = nn.Dropout(dropout)
        self.rl1 = nn.ReLU()
        
        #classification layer:  classifies the 100 values into num_classes values
        self.fc2 = nn.Linear(100, num_classes)       
    
    ''' This is my first, more complicated network.  Udacity servers couldn't train it
        #conv layer 1: gets 3x224x224, puts out 8 feature maps of 112*112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 8, kernel_size = 3, padding=1) #224x224x8
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2) #112x112x8
        
        #conv layer 2: gets 8x112x112, puts out 16 feature maps of 56*56
        self.conv2 = nn.Conv2d(in_channels=8, out_channels = 16, kernel_size = 3, padding=1) #112x112x16
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2) #56x56x16
        
        #conv layer 3: gets 16 56x56 feature maps, puts out 32 feature maps of 28*28
        self.conv3 = nn.Conv2d(in_channels=16, out_channels = 32, kernel_size = 3, padding=1) #56x56x32
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2) #28x28x32
        
        #conv layer 4: gets 32 28x28 feature maps, puts out 64 feature maps of 14*14
        self.conv4 = nn.Conv2d(in_channels=32, out_channels = 64, kernel_size = 3, padding=1) #28x28x64
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2) #14x14x64
        
        #conv layer 5: gets 64 14x14 feature maps, puts out 128 feature maps of 7*7
        self.conv5 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, padding=1) #14x14x128
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2,2) #7x7x128
        
        #linear layer #1:  turns the 128 7x7 feature maps (6272 values) into a 1000 values 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 1000)
        self.dp1 = nn.Dropout(dropout)
        self.rl1 = nn.ReLU()
        
        #linear layer #2:  turns thes 1000 values into 500 values
        self.fc2 = nn.Linear(1000, 500)
        self.dp2 = nn.Dropout(dropout)
        self.rl2 = nn.ReLU()
        
        #classification layer:  classifies the 500 values into num_classes values
        self.fc3 = nn.Linear(500, num_classes)       
       '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x=self.relu1(self.pool1(self.conv1(x)))
        x=self.relu2(self.pool2(self.conv2(x)))
        x=self.relu3(self.pool3(self.conv3(x)))
        x=self.relu4(self.pool4(self.conv4(x)))
        x=self.relu5(self.pool5(self.conv5(x)))
        
        x = self.flatten(x)
        
        x = self.rl1(self.dp1(self.fc1(x)))
        
        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

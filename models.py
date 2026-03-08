import torch
import torch.nn as nn


class ClassficationLoss(nn.Module):
    '''A wrapper for the cross-entropy loss used in classification tasks.'''
    def __init__(self):
        super(ClassficationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.criterion(logits, labels)
    

class FaceDiseaseCNN(nn.Module):
    '''A simple CNN architecture for face disease classification.'''
    
    def __init__(self, in_channels=3, num_classes=5, block_channels=[64, 128, 256], channels_l0=64, stride=1, **kwargs):
        # Because we are working with 3-channel images, we set in_channels to 3 by default
        super(FaceDiseaseCNN, self).__init__()
        
        # Blow number of channels in first block to channels_l0 and use stride of 2 to reduce spatial dimensions
        cnn_layers = [
            nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=5),  # First layer with stride 2
            nn.BatchNorm2d(channels_l0),
            nn.ReLU()
        ]
        current_channels = channels_l0
        for ch in block_channels:
            cnn_layers.append(nn.Conv2d(current_channels, ch, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.BatchNorm2d(ch))
            cnn_layers.append(nn.ReLU())
            current_channels = ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to reduce spatial dimensions to 1x1
        self.fc = nn.Linear(current_channels, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the CNN
        x = self.fc(x)
        return x
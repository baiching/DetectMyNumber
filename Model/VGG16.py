import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class VGG16Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self._two_conv_layer(1, 64, 64)
        self.block2 = self._two_conv_layer(64, 128, 128)
        self.block3 = self._three_conv_layer(128, 256, 256, 256)
        self.block4 = self._three_conv_layer(256, 512, 512, 512)
        self.prediction = nn.Sequential(
            Flatten(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.prediction(self.block4(self.block3(self.block2(self.block1(x)))))

        return out
        
    def _two_conv_layer(self, in_channel, out_channel1, out_channel2):
        """
        It consists two consecutive Conv layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(True),
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def _three_conv_layer(self, in_channel, out_channel1, out_channel2, out_channel3):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(True),
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(True),
            nn.Conv2d(out_channel2, out_channel3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def _four_conv_layer(self, in_channel, out_channel1, out_channel2, out_channel3, out_channel4):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel1),
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel2, out_channel3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channel3, out_channel4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
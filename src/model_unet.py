import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv1d -> BatchNorm1d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        # 1D transposed convolution to double the spatial dimension
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # after concatenation, channels will be (in_channels // 2) + skip_channels = in_channels
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        # x1 is from previous decoder layer, x2 is the skip connection from the encoder
        x1 = self.up(x1)
        
        # Dimensions: [Batch, Channels, Length]
        # Calculate padding if sizes do not perfectly match (though 1000->500->250->125 works perfectly)
        diff = x2.size()[2] - x1.size()[2]
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
            
        x = torch.cat([x2, x1], dim=1) # concatenate along channel dimension
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet1D, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)         # Length: 1000
        self.down1 = Down(64, 128)                     # Length: 500
        self.down2 = Down(128, 256)                    # Length: 250
        self.down3 = Down(256, 512)                    # Length: 125 (Bottleneck)
        
        # Decoder
        # Up gets (in_channels, out_channels)
        # in_channels=512 -> ConvTranspose outputs 256 -> cat with 256 -> 512 -> DoubleConv to 256
        self.up1 = Up(512, 256)                        # Length: 250
        self.up2 = Up(256, 128)                        # Length: 500
        self.up3 = Up(128, 64)                         # Length: 1000
        
        # Regression Head
        self.outc = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck
        
        # Decoder path with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output prediction map (Linear activation, [Batch, 1, 1000])
        output = self.outc(x)
        return output

import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, do_batch_norm=True):
        super(Conv2dBlock, self).__init__()
        self.do_batch_norm = do_batch_norm

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if do_batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if do_batch_norm else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.do_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.do_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        return x


class RoadSegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        num_filters=16,
        dropout=0.1,
        do_batch_norm=True,
    ):
        super().__init__()

        self.enc1 = Conv2dBlock(
            in_channels, num_filters * 1, do_batch_norm=do_batch_norm
        )
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout)

        self.enc2 = Conv2dBlock(
            num_filters * 1, num_filters * 2, do_batch_norm=do_batch_norm
        )
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout)

        self.enc3 = Conv2dBlock(
            num_filters * 2, num_filters * 4, do_batch_norm=do_batch_norm
        )
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout)

        self.enc4 = Conv2dBlock(
            num_filters * 4, num_filters * 8, do_batch_norm=do_batch_norm
        )
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(dropout)

        self.center = Conv2dBlock(
            num_filters * 8, num_filters * 16, do_batch_norm=do_batch_norm
        )

        self.up6 = nn.ConvTranspose2d(
            num_filters * 16,
            num_filters * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.dec6 = Conv2dBlock(
            num_filters * 16, num_filters * 8, do_batch_norm=do_batch_norm
        )
        self.drop6 = nn.Dropout(dropout)

        self.up7 = nn.ConvTranspose2d(
            num_filters * 8,
            num_filters * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.dec7 = Conv2dBlock(
            num_filters * 8, num_filters * 4, do_batch_norm=do_batch_norm
        )
        self.drop7 = nn.Dropout(dropout)

        self.up8 = nn.ConvTranspose2d(
            num_filters * 4,
            num_filters * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.dec8 = Conv2dBlock(
            num_filters * 4, num_filters * 2, do_batch_norm=do_batch_norm
        )
        self.drop8 = nn.Dropout(dropout)

        self.up9 = nn.ConvTranspose2d(
            num_filters * 2,
            num_filters * 1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.dec9 = Conv2dBlock(
            num_filters * 2, num_filters * 1, do_batch_norm=do_batch_norm
        )
        self.drop9 = nn.Dropout(dropout)

        self.final = nn.Conv2d(num_filters * 1, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.enc4(p3)
        p4 = self.drop4(self.pool4(c4))

        c5 = self.center(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.dec6(self.drop6(u6))

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.dec7(self.drop7(u7))

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.dec8(self.drop8(u8))

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.dec9(self.drop9(u9))

        logits = self.final(c9)
        return logits

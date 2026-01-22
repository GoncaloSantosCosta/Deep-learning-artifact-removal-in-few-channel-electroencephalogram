import torch.nn as nn
    
## RESNET 1D with Squeeze-and-Excitation block   
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class SE_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_residual=True, dropout_rate=0.1,reduction=16):
        super(SE_ResBlock, self).__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2= nn.Dropout(dropout_rate)

        # Squeeze-and-Excitation block
        reduction = max(4, out_channels // 16)
        self.se = SELayer1D(out_channels, reduction)

        if self.use_residual and (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.LeakyReLU(0.2)
       

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout1(out)

        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out = self.se(out)

        if self.use_residual:
            out += residual

        out = self.activation(out)
        return out

class SE_ResNet1D(nn.Module):
    def __init__(self, input_channels, num_blocks, channels, kernel_sizes, reduction, use_residual=True, dropout_rate=0.1):
        super(SE_ResNet1D, self).__init__()
        assert num_blocks == len(channels) == len(kernel_sizes), \
            "Length of channels and kernel_sizes must match num_blocks"

        self.blocks = nn.ModuleList()
        in_ch = input_channels

        for i in range(num_blocks):
            out_ch = channels[i]
            k = kernel_sizes[i]
            block = SE_ResBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                use_residual=use_residual,
                dropout_rate=dropout_rate,
                reduction=reduction
            )
            self.blocks.append(block)
            in_ch = out_ch

        self.final = nn.Conv1d(in_ch, input_channels, kernel_size=3, padding='same')

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        return x
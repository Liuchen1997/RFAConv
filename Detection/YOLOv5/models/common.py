# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""


class CBAMConv(nn.Module):
    def __init__(self, channel, out_channel, kernel_size, stride, reduction=16, spatial_kernel=7):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.spatital = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                  padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.spatital(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return self.conv(x)


class CAMConv(nn.Module):
    def __init__(self, channel, out_channel, kernel_size, stride, reduction=16, spatial_kernel=7):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return self.conv(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAConv(nn.Module):
    def __init__(self, inp, oup,kernel_size,stride, reduction=32):
        super(CAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,padding=kernel_size//2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.ReLU())
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return self.conv(out)
    
    

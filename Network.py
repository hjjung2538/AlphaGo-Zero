import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, block, blocks, size):
        super(Net, self).__init__()
        self.size = size
        self.conv = nn.Conv2d(in_channels=17, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.layer = self.make_layer(block, 256, blocks)
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*size**2, size ** 2 + 1)
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(size**2, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()

    def make_layer(self, block, out_channels, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(out_channels, out_channels, stride=1, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer(out)

        p_logits = self.policy_conv(out)
        p_logits = self.policy_bn(p_logits)
        p_logits = p_logits.view(-1, 2*self.size**2)
        p_logits = self.relu(p_logits)
        p_logits = self.policy_fc(p_logits)

        v = self.value_conv(out)
        v = self.value_bn(v)
        v = v.view(-1, self.size**2)
        v = self.relu(v)
        v = self.value_fc1(v)
        v = self.relu(v)
        v = self.value_fc2(v)
        v = self.value_tanh(v)

        # print(p_logits)
        # print(F.softmax(p_logits))
        # print(v)
        return p_logits, v
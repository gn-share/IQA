import torch as torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from transformer import Transformer
from einops.layers.torch import Rearrange
import numpy as np
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BaseModel(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super(BaseModel, self).__init__()
        self.res = resnet50_backbone(pretrained=pretrained)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hyperInChn = 112
        feature_size = 7
        self.feature_size = feature_size
        # Conv layers for resnet output features
        self.fuse1 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(4096, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048+1024+512+256, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )
        self.target_in_size = 224
        self.f1 = 112
        self.f2 = 56
        self.f3 = 28
        self.f4 = 14

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(
            self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(
            self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 5:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, x):
        x, y = self.res(x)
        y = torch.cat([self.fuse1(y[0]), self.fuse2(
            y[1]), self.fuse3(y[2]), y[3]], dim=1)
        self.conv1(y)
        hyper_in_feat = self.conv1(
            y).view(-1, self.hyperInChn, self.feature_size, self.feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(
            hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(
            hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(
            hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(
            hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(
            self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return x.view(-1, self.target_in_size, 1, 1), out


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2)//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None]*a[None, :])
        g = g/torch.sum(g)
        self.register_buffer(
            'filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride,
                       padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()


class LearningPositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, width, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, width*width)
        self.width = width
        self.seq_length = max_position_embeddings

    def forward(self, x):
        positions = torch.arange(
            self.seq_length, dtype=torch.long).to(x.device)
        position_embeddings = self.position_embeddings(positions)
        positions_broadcast_shape = [
            1, self.seq_length, self.width, self.width]
        position_embeddings = position_embeddings.view(
            *positions_broadcast_shape)
        position_embeddings
        return position_embeddings


class CompressModel(nn.Module):

    def __init__(self, channel=256, width_height=112, size=7, nhead=8, num_encoder_layers=2, dim_feedforward=64, dropout=0.1,
                 activation="relu", normalize_before=True
                 ) -> None:
        super(CompressModel, self).__init__()
        self.d_model = int(channel*(width_height/size)*(width_height/size))
        self.l2 = L2pooling(channels=self.d_model)
        self.transformer = Transformer(
            self.d_model, nhead, num_encoder_layers,
            dim_feedforward, dropout,
            activation, normalize_before
        )
        self.position_embedding = LearningPositionEmbedding(
            self.d_model, width=size)

        self.change = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) p1 p2', p1=size, p2=size)
        )
        self.rechange = nn.Sequential(
            Rearrange('b (c h w) p1 p2 -> b c (h p1) (w p2)', c=channel,
                      h=int(width_height/size), w=int(width_height/size))
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # 将图像转换为需要的格式
        y = self.change(x)
        y = self.l2(y)
        y1 = self.transformer(y, self.position_embedding(x))
        y = self.rechange(y1)

        return self.sig(y)*x, y1


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, 224)

        self.c1 = CompressModel(256, 56, 28)
        self.c2 = CompressModel(512, 28, 14)
        self.c3 = CompressModel(1024, 14)
        self.c4 = CompressModel(2048, 7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, images):
        x = images[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x, lda1 = self.c1(x)
        x = self.layer2(x)
        x, lda2 = self.c2(x)
        x = self.layer3(x)
        x, lda3 = self.c3(x)
        x = self.layer4(x)
        x, lda4 = self.c4(x)
        # lda_4 = self.lda4_fc(torch.cat([self.lda4_pool(x).view(x.size(0), -1),lda1,lda2,lda3,lda4],dim=1))
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))
        return lda_4, [lda1, lda2, lda3, lda4]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet50_backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model

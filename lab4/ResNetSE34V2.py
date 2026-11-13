# Основа архитектуры заимствована из проекта voxceleb_trainer:
# https://github.com/clovaai/voxceleb_trainer
# Модуль создаёт ResNetSE для распознавания диктора.

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from ResNetBlocks import SEBasicBlock
from preproc import PreEmphasis
from constants import (
    ATTENTION_HIDDEN_SIZE,
    EMBEDDING_SIZE,
    ENCODER_ATTENTIVE_STATISTICS_POOLING,
    ENCODER_SELF_ATTENTIVE_POOLING,
    ERROR_UNKNOWN_ENCODER,
    FEATURE_EPSILON,
    FFT_SIZE,
    HOP_LENGTH,
    MEL_FILTERS,
    MODEL_INITIALIZATION_MESSAGE,
    RESNET_FILTERS,
    RESNET_LAYERS,
    SAMPLE_RATE,
    STATISTICS_EPSILON,
    WINDOW_LENGTH,
)


class ResNetSE(nn.Module):
    """ResNetSE-модель для построения дикторского эмбеддинга."""

    def __init__(self, block, layers, num_filters, nOut, encoder_type=ENCODER_ATTENTIVE_STATISTICS_POOLING, n_mels=MEL_FILTERS, log_input=True, **kwargs):
        
        super(ResNetSE, self).__init__()

        print(
            MODEL_INITIALIZATION_MESSAGE.format(
                embedding_size=nOut,
                encoder_type=encoder_type,
            )
        )
        
        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.bn1   = nn.BatchNorm2d(num_filters[0])
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=FFT_SIZE,
                win_length=WINDOW_LENGTH,
                hop_length=HOP_LENGTH,
                window_fn=torch.hamming_window,
                n_mels=n_mels,
            ),
        )

        outmap_size = (self.n_mels + 7)//8
        frame_feature_size = num_filters[3]*block.expansion*outmap_size

        self.attention = nn.Sequential(
            nn.Conv1d(frame_feature_size, ATTENTION_HIDDEN_SIZE, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(ATTENTION_HIDDEN_SIZE),
            nn.Conv1d(ATTENTION_HIDDEN_SIZE, frame_feature_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == ENCODER_SELF_ATTENTIVE_POOLING:
            out_dim = frame_feature_size
        
        elif self.encoder_type == ENCODER_ATTENTIVE_STATISTICS_POOLING:
            out_dim = frame_feature_size*2
        
        else:
            raise ValueError(ERROR_UNKNOWN_ENCODER.format(encoder_type=encoder_type))

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes*block.expansion),
                                       )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        
        return out

    def forward(self, x):

        with torch.no_grad():
            x = self.torchfb(x.float()) + FEATURE_EPSILON

            if self.log_input: x = x.log()

            x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == ENCODER_SELF_ATTENTIVE_POOLING:
            x = torch.sum(x*w, dim=2)
        
        elif self.encoder_type == ENCODER_ATTENTIVE_STATISTICS_POOLING:
            mu = torch.sum(x*w, dim=2)
            sg = torch.sqrt((torch.sum((x**2)*w, dim=2) - mu**2).clamp(min=STATISTICS_EPSILON))
            x  = torch.cat((mu,sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

def MainModel(nOut=EMBEDDING_SIZE, **kwargs):
    """Создать основную модель распознавания диктора."""
    
    model = ResNetSE(SEBasicBlock, RESNET_LAYERS, RESNET_FILTERS, nOut, **kwargs)
    
    return model

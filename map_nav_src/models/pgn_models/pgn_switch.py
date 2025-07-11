import torch
from torch import nn
import torchvision
from .resnet import ResNet

def pgn_switch(
        model_type,
        out_features,
        **kwargs
) -> torch.nn.Module:
    if model_type == 'resnet18':
        return generate_resnet18(out_features, **kwargs)
    elif model_type == 'resnet10':
        return create_small_resnet(out_features, **kwargs)
    elif model_type == 'MLP':
        return create_MLP(out_features)

def create_MLP(out_feature):
    mlp = Critic(out_feature)
    return mlp

def generate_resnet18(out_features, pretrained_pgn, **kwargs):
    resnet = torchvision.models.resnet18(pretrained=pretrained_pgn)
    resnet.fc = torch.nn.Linear(
        in_features=512,
        out_features=out_features
    )
    return resnet


def create_small_resnet(out_features,
                        proj_type,
                        blocks_per_group,
                        initial_channels,
                        nr_groups,
                        init_max_pool=False,
                        **kwargs):
    resnet = ResNet(
        num_classes=out_features,
        proj_type=proj_type,
        num_blocks=[blocks_per_group] * nr_groups,
        # c_hidden=[initial_channels * (2 ** power) for power in range(nr_groups)],
        init_max_pool=init_max_pool,
    )
    return resnet

class Critic(nn.Module):
    def __init__(self, out_feature):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, out_feature),
        )
        self._init_params()
    
    def _init_params(self):
        for m in self.state2value.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight)

    def forward(self, state):
        return self.state2value(state)
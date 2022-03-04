import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import timm

class CreateModel(nn.Module):
    def __init__(self, config , pretrained : bool = True, Multi_Sample_Dropout : bool = True):
        super(CreateModel, self).__init__()
        self.Multi_Sample_Dropout = Multi_Sample_Dropout
        if self.Multi_Sample_Dropout:
            self.model = timm.create_model(config.timm_model_name, pretrained = pretrained)
            out_features = self.model.head.fc.out_features
            self.output_layer = nn.Linear(in_features = out_features, out_features = config.num_classes, bias=True)
            self.dropouts = nn.ModuleList([
                    nn.Dropout(0.7) for _ in range(5)])
        else:
            self.model = timm.create_model(config.timm_model_name, pretrained = pretrained, num_classes = config.num_classes)

    def forward(self, img):
        if self.Multi_Sample_Dropout:
            feat = self.model(img)
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = self.output_layer(dropout(feat))
                else:
                    out += self.output_layer(dropout(feat))
            else:
                out /= len(self.dropouts)
        else:
            out = self.model(img)
        return out
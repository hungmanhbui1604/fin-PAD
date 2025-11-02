import torch
import torch.nn as nn
import torchvision.models as models

class DualModel(nn.Module):
    def __init__(self, num_classes):
        super(DualModel, self).__init__()
        
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16.avgpool = nn.Identity()
        self.vgg16.classifier = nn.Identity()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.avgpool = nn.Identity()
        self.resnet50.fc = nn.Identity()
        
        combined_features = 25088 + 100352
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        vgg_features = self.vgg16(x)            # [B, 25088]
        resnet_features = self.resnet50(x)      # [B, 100352]
        combined_features = torch.cat([vgg_features, resnet_features], dim=1)  # [B, 125440] 
        logits = self.classifier(combined_features)  # [B, num_classes]
        
        return logits
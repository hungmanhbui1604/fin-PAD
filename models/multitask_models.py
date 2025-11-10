import torch
import torch.nn as nn
import torchvision.models as models


class MultitaskModel(nn.Module):
    def __init__(self, feature_extractor, num_material_classes):  # Only material classes, no "real" class
        super().__init__()

        # Load feature extractor
        self.feature_extractor = feature_extractor
        
        # Feature dimension after feature extractor
        self.feature_dim = feature_extractor.feature_dim
        
        # Binary classification head (spoof vs real)
        self.spoof_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Material classification head (only for spoof images)
        self.material_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_material_classes)
        )
    
    def forward(self, x):
        # Extract features using backbone
        features = self.feature_extractor(x)
        
        # Get spoof prediction (binary)
        spoof_output = self.spoof_classifier(features)
        
        # Get material prediction (multiclass - only meaningful if spoof=1)
        material_output = self.material_classifier(features)
        
        return spoof_output, material_output
    

class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained MobileNetV2 with latest weights parameter
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Extract features part (excluding classifier)
        self.features = mobilenet.features
        
        # Global average pooling to get fixed-size features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension after feature extractor (MobileNetV2's last channel size)
        self.feature_dim = 1280
        
    def forward(self, x):
        # Extract features using backbone
        x = self.features(x)
        
        # Global average pooling to get fixed-size features
        x = self.avg_pool(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        return x
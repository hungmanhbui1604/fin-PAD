import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


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


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet50 with latest weights parameter
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Extract features part (excluding classifier)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension after feature extractor (ResNet50's last channel size)
        self.feature_dim = 2048
        
    def forward(self, x):
        # Extract features using backbone
        x = self.features(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        return x


class EfficientNetB0Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained EfficientNet-B0 with latest weights parameter
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Extract features part (excluding classifier)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Feature dimension after feature extractor (EfficientNet-B0's last channel size)
        self.feature_dim = 1280
        
    def forward(self, x):
        # Extract features using backbone
        x = self.features(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        return x


class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool

        # Feature dimension after feature extractor (VGG16's flattened features)
        self.feature_dim = 25088

    def forward(self, x):
        # Extract features using VGG16 backbone
        x = self.features(x)
        x = self.avgpool(x)

        # Flatten the features
        x = torch.flatten(x, 1)

        return x


class DualModelBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Exact same setup as DualModel in classic_models.py
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16.avgpool = nn.Identity()
        self.vgg16.classifier = nn.Identity()

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.avgpool = nn.Identity()
        self.resnet50.fc = nn.Identity()

        # Same feature dimensions as DualModel
        self.feature_dim = 25088 + 100352

    def forward(self, x):
        # Exact same feature extraction as DualModel in classic_models.py
        vgg_features = self.vgg16(x)            # [B, 25088]
        resnet_features = self.resnet50(x)      # [B, 100352]
        combined_features = torch.cat([vgg_features, resnet_features], dim=1)  # [B, 125440]

        return combined_features


def get_model(
    backbone_name,
    num_material_classes,
    spoof_criterion_type='bce_with_logits', 
    material_criterion_type='cross_entropy',
    optimizer_type='adam',
    learning_rate=1e-3,
    weight_decay=1e-4,
    scheduler_type='cosine_annealing',
    num_epochs=100,
):
    
    if backbone_name.lower() == 'mobilenetv2':
        backbone = MobileNetV2Backbone()
    elif backbone_name.lower() == 'resnet50':
        backbone = ResNet50Backbone()
    elif backbone_name.lower() == 'efficientnetb0':
        backbone = EfficientNetB0Backbone()
    elif backbone_name.lower() == 'vgg16':
        backbone = VGG16Backbone()
    elif backbone_name.lower() == 'dual':
        backbone = DualModelBackbone()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Supported backbones: 'mobilenetv2', 'resnet50', 'efficientnetb0', 'vgg16', 'dual'")
    
    # Create the multitask model
    model = MultitaskModel(backbone, num_material_classes)
    
    # Define spoof criterion based on type
    if spoof_criterion_type.lower() == 'bce_with_logits':
        spoof_criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported spoof criterion type: {spoof_criterion_type}. Supported types: 'bce_with_logits'")
    
    # Define material criterion based on type
    if material_criterion_type.lower() == 'cross_entropy':
        material_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported material criterion type: {material_criterion_type}. Supported types: 'cross_entropy'")
    
    # Create optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: 'adam'")
    
    # Define scheduler based on type
    if scheduler_type.lower() == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=1e-6)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Supported types: 'cosine_annealing'")
    
    return model, spoof_criterion, material_criterion, optimizer, scheduler
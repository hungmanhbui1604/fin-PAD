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
        logits = self.classifier(combined_features) # [B, num_classes]
        
        return logits


class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16.avgpool = nn.Identity()
        self.vgg16.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.vgg16(x)            # [B, 25088]
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits


class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.avgpool = nn.Identity()
        self.resnet50.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(100352, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.resnet50(x)            # [B, 100352]
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits


def get_model(
        model_name, 
        num_classes, 
        criterion_type='bce_with_logits', 
        optimizer_type='adam',
        learning_rate=1e-3, 
        weight_decay=1e-4, 
        scheduler_type='cosine_annealing', 
        num_epochs=100):
    # Initialize model
    if model_name.lower() == 'dual':
        model = DualModel(num_classes)
    elif model_name.lower() == 'vgg16':
        model = VGG16Model(num_classes)
    elif model_name.lower() == 'resnet50':
        model = ResNet50Model(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: 'dual', 'vgg16', 'resnet50'")
    
    # Initialize criterion
    if criterion_type == 'bce_with_logits':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")
    
    # Initialize optimizer
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Initialize scheduler
    if scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return model, criterion, optimizer, scheduler

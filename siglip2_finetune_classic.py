import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import yaml
import argparse
import os
import copy

from transformers import AutoProcessor, SiglipVisionModel
from data_utils.data_loaders import get_classic_data_loaders
from data_utils.transforms import get_transforms

class SigLIP2Finetuner(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        
        # Load only the vision transformer part of SigLIP
        print(f"Loading Vision Backbone: {model_name}")
        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        
        # Get embedding dimension (usually 1152 for SO400M or 768 for smaller variants)
        self.embed_dim = self.backbone.config.hidden_size
        
        # Classification Head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Optional: Freeze the backbone to train only the head first
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Training classifier head only.")
        else:
            print("Backbone unfrozen. Full fine-tuning enabled.")

    def forward(self, pixel_values):
        # Pass through Vision Transformer
        outputs = self.backbone(pixel_values=pixel_values)
        
        # SigLIP usually uses the pooled output or the first token [CLS] equivalent
        # SiglipVisionModel outputs usually have 'pooler_output'
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            # Fallback: take the first token (CLS) from last_hidden_state
            features = outputs.last_hidden_state[:, 0, :]
            
        # Pass through classifier
        logits = self.classifier(features)
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy for progress bar
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        current_acc = accuracy_score(all_labels[-100:], all_preds[-100:]) # Moving average
        pbar.set_postfix({'loss': loss.item(), 'acc': f"{current_acc:.3f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description='Fine-tune SigLIP2')
    parser.add_argument('--config', type=str, default='./configs/siglip2_finetune.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)

    # 1. Setup Data Loaders
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader, label_map = get_classic_data_loaders(config)

    num_classes = len(label_map)
    print(f"Detected {num_classes} classes: {label_map}")

    # 2. Initialize Model
    model = SigLIP2Finetuner(
        model_name=config['MODEL_NAME'], 
        num_classes=num_classes,
        freeze_backbone=False # Set True if you only want to train the head
    ).to(device)

    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates (optional but recommended):
    # Lower LR for backbone, Higher LR for head
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': config['LR'] * 0.1}, # 10x smaller LR for backbone
        {'params': model.classifier.parameters(), 'lr': config['LR']}
    ], lr=config['LR'], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=config['EPOCHS'])
    
    best_val_acc = 0.0

    # 4. Training Loop
    print("\nStarting Fine-tuning...")
    for epoch in range(1, config['EPOCHS'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{config['EPOCHS']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config['OUTPUT_DIR'], 'best_siglip_finetuned.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
import torch
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, train_loader, spoof_criterion, material_criterion, optimizer, device, config):
    model.train()
    running_spoof_loss = 0.0
    running_material_loss = 0.0
    running_total_loss = 0.0
    running_correct_spoof = 0
    running_correct_material = 0
    total_samples = 0
    total_spoof_samples = 0
    
    for imgs, labels in tqdm(train_loader, desc="train"):
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.float)
        
        spoof_labels = (labels > 0).float().unsqueeze(1)
        material_labels = (labels - 1).long()
        
        optimizer.zero_grad()
        
        spoof_outputs, material_outputs = model(imgs)
        
        spoof_loss = spoof_criterion(spoof_outputs, spoof_labels)
        
        spoof_indices = (spoof_labels.squeeze() == 1).nonzero(as_tuple=True)[0]
        if len(spoof_indices) > 0:
            spoof_material_outputs = material_outputs[spoof_indices]
            spoof_material_labels = material_labels[spoof_indices]
            material_loss = material_criterion(spoof_material_outputs, spoof_material_labels)
        else:
            material_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        total_loss = config['SPOOF_WEIGHT'] * spoof_loss + config['MATERIAL_WEIGHT'] * material_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_spoof_loss += spoof_loss.item()
        running_material_loss += material_loss.item()
        running_total_loss += total_loss.item()
        
        spoof_preds = (spoof_outputs > config['THRESHOLD']).float()
        running_correct_spoof += (spoof_preds == spoof_labels).sum().item()
        
        if len(spoof_indices) > 0:
            spoof_material_preds = torch.argmax(spoof_material_outputs, dim=1)
            running_correct_material += (spoof_material_preds == spoof_material_labels).sum().item()
            total_spoof_samples += len(spoof_indices)
        
        total_samples += imgs.size(0)
    
    epoch_spoof_loss = running_spoof_loss / len(train_loader)
    epoch_material_loss = running_material_loss / len(train_loader)
    epoch_total_loss = running_total_loss / len(train_loader)
    epoch_spoof_acc = (running_correct_spoof / total_samples) * 100.0
    epoch_material_acc = (running_correct_material / total_spoof_samples) * 100.0 if total_spoof_samples > 0 else 0.0
    
    print(f'Train Loss: Total=[{epoch_total_loss:.4f}] Spoof=[{epoch_spoof_loss:.4f}] Material=[{epoch_material_loss:.4f}]')
    print(f'Train Acc: Spoof=[{epoch_spoof_acc:.2f}] Material=[{epoch_material_acc:.2f}]')

def validate_one_epoch(model, val_loader, spoof_criterion, material_criterion, device, config):
    model.eval()
    val_spoof_loss = 0.0
    val_material_loss = 0.0
    val_total_loss = 0.0
    val_correct_spoof = 0
    val_correct_material = 0
    val_total_samples = 0
    val_spoof_samples_total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="val"):
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float)
            
            spoof_labels = (labels > 0).float().unsqueeze(1)
            material_labels = (labels - 1).long()
            
            spoof_outputs, material_outputs = model(imgs)
            
            spoof_loss = spoof_criterion(spoof_outputs, spoof_labels)
            
            spoof_indices = (spoof_labels.squeeze() == 1).nonzero(as_tuple=True)[0]
            if len(spoof_indices) > 0:
                spoof_material_outputs = material_outputs[spoof_indices]
                spoof_material_labels = material_labels[spoof_indices]
                material_loss = material_criterion(spoof_material_outputs, spoof_material_labels)
            else:
                material_loss = torch.tensor(0.0, device=device)
            
            total_loss = config['SPOOF_WEIGHT'] * spoof_loss + config['MATERIAL_WEIGHT'] * material_loss
            
            val_spoof_loss += spoof_loss.item()
            val_material_loss += material_loss.item()
            val_total_loss += total_loss.item()
            
            spoof_preds = (spoof_outputs > config['THRESHOLD']).float()
            val_correct_spoof += (spoof_preds == spoof_labels).sum().item()
            
            if len(spoof_indices) > 0:
                spoof_material_preds = torch.argmax(spoof_material_outputs, dim=1)
                val_correct_material += (spoof_material_preds == spoof_material_labels).sum().item()
            
            val_total_samples += imgs.size(0)
            val_spoof_samples_total += spoof_labels.sum().item()
    
    val_epoch_spoof_loss = val_spoof_loss / len(val_loader)
    val_epoch_material_loss = val_material_loss / len(val_loader)
    val_epoch_total_loss = val_total_loss / len(val_loader)
    val_epoch_spoof_acc = (val_correct_spoof / val_total_samples) * 100.0
    val_epoch_material_acc = (val_correct_material / val_spoof_samples_total) * 100.0 if val_spoof_samples_total > 0 else 0.0
    
    print(f'Val Loss: Total=[{val_epoch_total_loss:.4f}] Spoof=[{val_epoch_spoof_loss:.4f}] Material=[{val_epoch_material_loss:.4f}]')
    print(f'Val Acc: Spoof=[{val_epoch_spoof_acc:.2f}] Material=[{val_epoch_material_acc:.2f}]')
    return val_epoch_total_loss

def test_one_epoch(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="test"):
            imgs, labels = imgs.to(device), labels.to(device, dtype=torch.float)

            spoof_labels = (labels > 0).long()
            
            spoof_outputs, material_outputs = model(imgs)

            probabilities = torch.sigmoid(spoof_outputs.squeeze(1))
            
            all_labels.extend(spoof_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    labels = np.array(all_labels).astype(int)
    probabilities = np.array(all_probabilities)
    return labels, probabilities


def train_one_epoch_binary(model, train_loader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    for imgs, labels in tqdm(train_loader, desc="train"):
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.float)  # Binary labels: 0 or 1
        
        optimizer.zero_grad()
        
        # Forward pass - model outputs logits
        outputs = model(imgs)
        
        # Calculate loss (BCEWithLogitsLoss expects float targets)
        loss = criterion(outputs.squeeze(1), labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics - convert logits to predictions using sigmoid
        predicted = (torch.sigmoid(outputs.squeeze(1)) > config['THRESHOLD']).long()
        
        running_correct += (predicted == labels).sum().item()
        total_samples += imgs.size(0)
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = (running_correct / total_samples) * 100.0
    
    print(f'Train Loss: [{epoch_loss:.4f}] Train Acc: [{epoch_acc:.2f}%]')


def validate_one_epoch_binary(model, val_loader, criterion, device, config):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total_samples = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="val"):
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float)  # Binary labels: 0 or 1
            
            # Forward pass - model outputs logits
            outputs = model(imgs)
            
            # Calculate loss (BCEWithLogitsLoss expects float targets)
            loss = criterion(outputs.squeeze(1), labels)
            
            # Statistics - convert logits to predictions using sigmoid
            predicted = (torch.sigmoid(outputs.squeeze(1)) > config['THRESHOLD']).long()
            
            val_correct += (predicted == labels).sum().item()
            val_total_samples += imgs.size(0)
            val_loss += loss.item()
    
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = (val_correct / val_total_samples) * 100.0
    
    print(f'Val Loss: [{val_epoch_loss:.4f}] Val Acc: [{val_epoch_acc:.2f}%]')
    return val_epoch_loss


def test_one_epoch_binary(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="test"):
            imgs, labels = imgs.to(device), labels.to(device, dtype=torch.long)

            # Forward pass - model outputs logits
            outputs = model(imgs)
            
            # Get probabilities for positive class (class 1) using sigmoid
            probabilities = torch.sigmoid(outputs.squeeze(1))
            
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    labels = np.array(all_labels).astype(int)
    probabilities = np.array(all_probabilities)
    return labels, probabilities

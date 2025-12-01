import torch
import os
import wandb
import numpy as np
from tqdm import tqdm
from data_utils.data_loaders import get_dataloader
from data_utils.transforms import get_transforms
from train_utils.metrics import find_optimal_threshold
from models.multitask_models import get_model as get_multitask_model
from models.classic_models import get_model as get_classic_model


class ClassicTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def epoch_train(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for imgs, labels in tqdm(train_loader, desc="train"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device, dtype=torch.float)  # Binary labels: 0 or 1

            optimizer.zero_grad()

            # Forward pass - model outputs logits
            outputs = model(imgs)

            # Calculate loss (BCEWithLogitsLoss expects float targets)
            loss = criterion(outputs.squeeze(1), labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics - convert logits to predictions using sigmoid
            predicted = (torch.sigmoid(outputs.squeeze(1)) > self.config['THRESHOLD']).long()

            running_correct += (predicted == labels).sum().item()
            total_samples += imgs.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (running_correct / total_samples) * 100.0

        print(f'Train Loss: [{epoch_loss:.4f}] Train Acc: [{epoch_acc:.2f}%]')

        # Return metrics for logging
        return {
            'train/loss': epoch_loss,
            'train/accuracy': epoch_acc
        }

    def epoch_validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="val"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float)  # Binary labels: 0 or 1

                # Forward pass - model outputs logits
                outputs = model(imgs)

                # Calculate loss (BCEWithLogitsLoss expects float targets)
                loss = criterion(outputs.squeeze(1), labels)

                # Statistics - convert logits to predictions using sigmoid
                predicted = (torch.sigmoid(outputs.squeeze(1)) > self.config['THRESHOLD']).long()

                val_correct += (predicted == labels).sum().item()
                val_total_samples += imgs.size(0)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = (val_correct / val_total_samples) * 100.0

        print(f'Val Loss: [{val_epoch_loss:.4f}] Val Acc: [{val_epoch_acc:.2f}%]')

        # Return metrics for logging
        return {
            'val/loss': val_epoch_loss,
            'val/accuracy': val_epoch_acc
        }

    def epoch_test(self, model, test_loader):
        model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="test"):
                imgs, labels = imgs.to(self.device), labels.to(self.device, dtype=torch.long)

                # Forward pass - model outputs logits
                outputs = model(imgs)

                # Get probabilities for positive class (class 1) using sigmoid
                probabilities = torch.sigmoid(outputs.squeeze(1))

                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        labels = np.array(all_labels).astype(int)
        probabilities = np.array(all_probabilities)
        return labels, probabilities


    def setup_wandb(self):
        """Initialize wandb for experiment tracking"""
        # Login to wandb using API key from config
        if self.config.get('WANDB_API_KEY'):
            wandb.login(key=self.config['WANDB_API_KEY'])

        wandb.init(
            project="FinPAD",
            name=f"classic_{self.config['MODEL_NAME']}_{self.config['YEAR']}_{self.config['TRAIN_SENSOR']}_{self.config['TEST_SENSOR']}",
            config=self.config,
            tags=["spoof_classification"]
        )

    def train(self, model, criterion, optimizer, scheduler, train_loader, val_loader):
        """Training loop for binary classification"""
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config['NUM_EPOCHS']):
            print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
            print('-' * 36)

            train_metrics = self.epoch_train(model, train_loader, criterion, optimizer)
            val_metrics = self.epoch_validate(model, val_loader, criterion)

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config['LEARNING_RATE'],
                **train_metrics,
                **val_metrics
            })

            # Save the best model based on validation loss
            if val_metrics['val/loss'] < best_val_loss:
                best_val_loss = val_metrics['val/loss']
                best_model_state = model.state_dict().copy()
                print(f"New best model found! Saving to {self.config['MODEL_SAVE_PATH']}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, self.config['MODEL_SAVE_PATH'])

                # Save model to wandb
                wandb.save(self.config['MODEL_SAVE_PATH'])

            scheduler.step()

        return best_model_state

    def test(self, model, test_loader):
        """Testing phase for binary classification"""
        labels, probabilities = self.epoch_test(model, test_loader)
        threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(labels, probabilities, based_on="ace")

        # Log final test results to wandb
        test_results = {
            'test/APCER': apcer * 100,
            'test/BPCER': bpcer * 100,
            'test/ACE': ace * 100,
            'test/Accuracy': accuracy * 100,
            'test/Accuracy_star': (1 - ace) * 100,
            'test/Threshold': threshold
        }
        wandb.log(test_results)

        print(f"APCER:      {apcer*100:.2f}%")
        print(f"BPCER:      {bpcer*100:.2f}%")
        print(f"ACE:        {ace*100:.2f}%")
        print(f"Accuracy:   {accuracy*100:.2f}%")
        print(f"Accuracy*:  {(1-ace)*100:.2f}%")

        return test_results

    def run(self, train_loader, val_loader, test_loader):
        """Main training pipeline"""
        # Create model save directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config['MODEL_SAVE_PATH']), exist_ok=True)

        # Initialize model and components from config
        model, criterion, optimizer, scheduler = get_classic_model(
            model_name=self.config['MODEL_NAME'],
            num_classes=1,  # Binary classification: real vs spoof
            criterion_type=self.config['CRITERION_TYPE'],
            optimizer_type=self.config['OPTIMIZER_TYPE'],
            learning_rate=self.config['LEARNING_RATE'],
            weight_decay=self.config['WEIGHT_DECAY'],
            scheduler_type=self.config['SCHEDULER_TYPE'],
            num_epochs=self.config['NUM_EPOCHS'],
        )

        # Initialize wandb
        self.setup_wandb()

        # Move model to device and handle multi-GPU
        model.to(self.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Training phase
        best_model_state = self.train(model, criterion, optimizer, scheduler, train_loader, val_loader)

        # Load the best model for testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Testing phase
        test_results = self.test(model, test_loader)

        # Finish wandb run
        wandb.finish()

        return test_results


class MultitaskTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def epoch_train(self, model, train_loader, spoof_criterion, material_criterion, optimizer):
        model.train()
        running_spoof_loss = 0.0
        running_material_loss = 0.0
        running_total_loss = 0.0
        running_correct_spoof = 0
        running_correct_material = 0
        total_samples = 0
        total_spoof_samples = 0

        for imgs, labels in tqdm(train_loader, desc="train"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device, dtype=torch.float)

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
                material_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            total_loss = self.config['SPOOF_WEIGHT'] * spoof_loss + self.config['MATERIAL_WEIGHT'] * material_loss

            total_loss.backward()
            optimizer.step()

            running_spoof_loss += spoof_loss.item()
            running_material_loss += material_loss.item()
            running_total_loss += total_loss.item()

            spoof_preds = (spoof_outputs > self.config['THRESHOLD']).float()
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

        # Return metrics for logging
        return {
            'train/total_loss': epoch_total_loss,
            'train/spoof_loss': epoch_spoof_loss,
            'train/material_loss': epoch_material_loss,
            'train/spoof_acc': epoch_spoof_acc,
            'train/material_acc': epoch_material_acc
        }

    def epoch_validate(self, model, val_loader, spoof_criterion, material_criterion):
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
                imgs = imgs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float)

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
                    material_loss = torch.tensor(0.0, device=self.device)

                total_loss = self.config['SPOOF_WEIGHT'] * spoof_loss + self.config['MATERIAL_WEIGHT'] * material_loss

                val_spoof_loss += spoof_loss.item()
                val_material_loss += material_loss.item()
                val_total_loss += total_loss.item()

                spoof_preds = (spoof_outputs > self.config['THRESHOLD']).float()
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

        # Return metrics for logging
        return {
            'val/total_loss': val_epoch_total_loss,
            'val/spoof_loss': val_epoch_spoof_loss,
            'val/material_loss': val_epoch_material_loss,
            'val/spoof_acc': val_epoch_spoof_acc,
            'val/material_acc': val_epoch_material_acc
        }

    def epoch_test(self, model, test_loader):
        model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="test"):
                imgs, labels = imgs.to(self.device), labels.to(self.device, dtype=torch.float)

                spoof_labels = (labels > 0).long()

                spoof_outputs, material_outputs = model(imgs)

                probabilities = torch.sigmoid(spoof_outputs.squeeze(1))

                all_labels.extend(spoof_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        labels = np.array(all_labels).astype(int)
        probabilities = np.array(all_probabilities)
        return labels, probabilities


    def setup_wandb(self):
        """Initialize wandb for experiment tracking"""
        # Login to wandb using API key from config
        if self.config.get('WANDB_API_KEY'):
            wandb.login(key=self.config['WANDB_API_KEY'])

        wandb.init(
            project="FinPAD",
            name=f"multitask_{self.config['BACKBONE_NAME']}_{self.config['YEAR']}_{self.config['TRAIN_SENSOR']}_{self.config['TEST_SENSOR']}",
            config=self.config,
            tags=["multitask_learning", "spoof_classification", "material_classification"]
        )

    def train(self, model, spoof_criterion, material_criterion, optimizer, scheduler, train_loader, val_loader):
        """Training loop for multitask learning"""
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config['NUM_EPOCHS']):
            print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
            print('-' * 36)

            train_metrics = self.epoch_train(model, train_loader, spoof_criterion, material_criterion, optimizer)
            val_metrics = self.epoch_validate(model, val_loader, spoof_criterion, material_criterion)

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config['LEARNING_RATE'],
                **train_metrics,
                **val_metrics
            })

            # Save the best model based on spoof validation loss
            if val_metrics['val/total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val/total_loss']
                best_model_state = model.state_dict().copy()
                print(f"New best model found! Saving to {self.config['MODEL_SAVE_PATH']}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, self.config['MODEL_SAVE_PATH'])

                # Save model to wandb
                wandb.save(self.config['MODEL_SAVE_PATH'])

            scheduler.step()

        return best_model_state

    def test(self, model, test_loader):
        """Testing phase for multitask learning"""
        labels, probabilities = self.epoch_test(model, test_loader)
        threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(labels, probabilities, based_on="ace")

        # Log final test results to wandb
        test_results = {
            'test/APCER': apcer * 100,
            'test/BPCER': bpcer * 100,
            'test/ACE': ace * 100,
            'test/Accuracy': accuracy * 100,
            'test/Accuracy_star': (1 - ace) * 100,
            'test/Threshold': threshold
        }
        wandb.log(test_results)

        print(f"APCER:      {apcer*100:.2f}%")
        print(f"BPCER:      {bpcer*100:.2f}%")
        print(f"ACE:        {ace*100:.2f}%")
        print(f"Accuracy:   {accuracy*100:.2f}%")
        print(f"Accuracy*:  {(1-ace)*100:.2f}%")

        return test_results

    def run(self, train_loader, val_loader, test_loader, train_label_map):
        """Main training pipeline"""
        # Create model save directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config['MODEL_SAVE_PATH']), exist_ok=True)

        # Initialize wandb
        self.setup_wandb()

        # Initialize model and components from config
        model, spoof_criterion, material_criterion, optimizer, scheduler = get_multitask_model(
            backbone_name=self.config['BACKBONE_NAME'],
            num_material_classes=len(train_label_map)-1,
            spoof_criterion_type=self.config['SPOOF_CRITERION_TYPE'],
            material_criterion_type=self.config['MATERIAL_CRITERION_TYPE'],
            optimizer_type=self.config['OPTIMIZER_TYPE'],
            learning_rate=self.config['LEARNING_RATE'],
            weight_decay=self.config['WEIGHT_DECAY'],
            scheduler_type=self.config['SCHEDULER_TYPE'],
            num_epochs=self.config['NUM_EPOCHS'],
        )

        # Move model to device and handle multi-GPU
        model.to(self.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Training phase
        best_model_state = self.train(model, spoof_criterion, material_criterion, optimizer, scheduler, train_loader, val_loader)

        # Load the best model for testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Testing phase
        test_results = self.test(model, test_loader)

        # Finish wandb run
        wandb.finish()

        return test_results
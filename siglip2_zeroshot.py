import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import yaml
import argparse
from transformers import AutoProcessor, AutoModel
from data_utils.data_loaders import create_data_loader
from data_utils.transforms import get_transforms
import os
from train_utils.metrics import compute_metrics


class SigLIP2ZeroShot:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device

        # Load SigLIP2 processor
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Load SigLIP2 model (with both vision and text components)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()


    def encode_images(self, image_batch):
        """Encode a batch of images using SigLIP2 vision model"""
        with torch.no_grad():
            inputs = {'pixel_values': image_batch.to(self.device)}

            # Get image features
            image_features = self.model.get_image_features(**inputs)

            # Normalize features
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_texts(self, texts):
        """Encode text prompts using SigLIP2 text model"""
        with torch.no_grad():
            # Process texts
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)

            # Get text features
            text_features = self.model.get_text_features(**inputs)

            # Normalize features
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def create_prompts(self, label_map):
        """Create text prompts for zero-shot classification"""
        prompts = []
        label_names = list(label_map.keys())

        # For binary classification (Live vs Spoof)
        if len(label_map) <= 2:
            prompts = [
                "a fingerprint image of a real live finger",
                "a fingerprint image of a fake spoof fingerprint"
            ]
        else:
            # For multiclass classification
            for label_name in label_names:
                if label_name.lower() == 'live':
                    prompts.append("a fingerprint image of a real live finger")
                else:
                    # More specific prompts for different spoof materials
                    prompts.append(f"a fingerprint image of a fake spoof fingerprint made of {label_name.lower()}")

        return prompts

    def predict_batch(self, images, text_features):
        """Perform zero-shot prediction on a batch of images"""
        # Encode images
        image_features = self.encode_images(images)

        # Compute similarity scores (dot product)
        similarity = torch.matmul(image_features, text_features.T)

        # Get predictions
        probs = torch.sigmoid(similarity)  # Use sigmoid for binary-like scores
        predictions = torch.argmax(similarity, dim=-1)

        return predictions.cpu().numpy(), probs.cpu().numpy()

    def evaluate_dataset(self, dataloader, label_map):
        """Evaluate on entire dataset"""
        all_predictions = []
        all_labels = []
        all_probs = []

        # Create text prompts and encode them once
        prompts = self.create_prompts(label_map)
        label_names = list(label_map.keys())
        print(f"\nText prompts:")
        for prompt, name in zip(prompts, label_names):
            print(f"{name}: {prompt}")

        text_features = self.encode_texts(prompts)

        # Track spoof vs live accuracy specifically
        binary_correct = 0
        binary_total = 0

        print("\nEvaluating dataset...")
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            # Get predictions for this batch
            pred_indices, probs = self.predict_batch(images, text_features)

            # Convert label indices to names for binary accuracy
            for i, (pred_idx, true_label) in enumerate(zip(pred_indices, labels)):
                pred_label_name = label_names[pred_idx]
                true_label_name = label_names[true_label]

                # Check if binary prediction is correct (Live vs Spoof)
                pred_is_spoof = not pred_label_name.lower() == 'live'
                true_is_spoof = not true_label_name.lower() == 'live'

                if pred_is_spoof == true_is_spoof:
                    binary_correct += 1
                binary_total += 1

            all_predictions.extend(pred_indices)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        binary_accuracy = binary_correct / binary_total if binary_total > 0 else 0

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # PAD metrics
        apcer, bpcer, ace, accuracy = compute_metrics(all_labels, all_predictions)

        # Print results
        print("\n=== Zero-Shot Evaluation Results ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Binary (Live/Spoof) Accuracy: {binary_accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")

        print("\n=== Per-Class Metrics ===")
        for i, label_name in enumerate(label_names):
            if i < len(precision_per_class):
                print(f"{label_name}:")
                print(f"  Precision: {precision_per_class[i]:.4f}")
                print(f"  Recall: {recall_per_class[i]:.4f}")
                print(f"  F1-Score: {f1_per_class[i]:.4f}")

        print("\n=== Confusion Matrix ===")
        print("Predicted →")
        print("Actual ↓")
        for i, label_name in enumerate(label_names):
            row = " ".join(f"{cm[i,j]:5d}" for j in range(len(label_names)))
            print(f"{label_name[:8]:8s} {row}")

        print("\n=== PAD Metrics ===")
        print(f"APCER: {apcer*100:.2f}%")
        print(f"BPCER: {bpcer*100:.2f}%")
        print(f"ACE: {ace*100:.2f}%")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Accuracy*: {(1-ace)*100:.2f}%")



def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description='Zero-shot evaluation with SigLIP2')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Initialize zero-shot model with config parameters
    print(f"Loading SigLIP2 model")
    model = SigLIP2ZeroShot(
        model_name=config['MODEL_NAME'],
        device=device
    )

    # Setup data loaders
    print("Setting up test data loader")
    test_loader, label_map = create_data_loader(
        year_path=config['YEAR_PATH'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        train=False,
        binary_class=config['BINARY_CLASS'],
        transform=get_transforms(config['TRANSFORM_TYPE']),
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS']
    )
    print(f"Test label map: {label_map}")

    # Evaluate on test set
    print("\nEvaluating")
    test_results = model.evaluate_dataset(test_loader, label_map)


if __name__ == "__main__":
    main()
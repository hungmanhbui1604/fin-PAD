import yaml
import argparse

from train_utils.trainers import MultitaskTrainer
from data_utils.data_loaders import setup_multitask_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='SigLIP 2 Multitask Learning for Fingerprint PAD')
    parser.add_argument('-c', '--config', type=str, default='./configs/siglip2_multitask_config.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("SigLIP 2 Multitask Learning for Fingerprint Presentation Attack Detection")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Backbone: {config['BACKBONE_NAME']}")
    print(f"Transform Type: {config['TRANSFORM_TYPE']}")
    print(f"Batch Size: {config['BATCH_SIZE']}")
    print(f"Learning Rate: {config['LEARNING_RATE']}")
    print("=" * 60)

    # Setup data loaders
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader, train_label_map = setup_multitask_data_loaders(config)

    # Initialize trainer and run training
    print("Initializing SigLIP 2 MultitaskTrainer...")
    trainer = MultitaskTrainer(config)

    print("Starting training...")
    test_results = trainer.run(train_loader, val_loader, test_loader, train_label_map)

    print("=" * 60)
    print("SigLIP 2 training completed!")
    print("Final test results:", test_results)
    print("=" * 60)
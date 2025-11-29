import yaml
import argparse

from train_utils.trainers import MultitaskTrainer
from data_utils.data_loaders import setup_multitask_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Multitask Learning')
    parser.add_argument('-c', '--config', type=str, default='./configs/multitask_config.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup data loaders
    train_loader, val_loader, test_loader, train_label_map = setup_multitask_data_loaders(config)

    # Initialize trainer and run training
    trainer = MultitaskTrainer(config)
    test_results = trainer.run(train_loader, val_loader, test_loader, train_label_map)
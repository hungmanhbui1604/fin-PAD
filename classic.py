import yaml
import argparse

from train_utils.trainers import ClassicTrainer
from data_utils.data_loaders import setup_classic_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Classification')
    parser.add_argument('-c', '--config', type=str, default='./configs/classic_config.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup data loaders
    train_loader, val_loader, test_loader = setup_classic_data_loaders(config)

    # Initialize trainer and run training
    trainer = ClassicTrainer(config)
    test_results = trainer.run(train_loader, val_loader, test_loader)
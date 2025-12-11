import os
from PIL import Image
import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from data_utils.transforms import get_transforms
from typing import Optional

class IntraSensorDataset(Dataset):
    def __init__(self, year_path, sensor, train=True, binary_class=True):
        self.samples = []
        self.label_map = {}
        
        if binary_class:
            self.label_map = {'Live': 0, 'Spoof': 1}
        else:
            self.label_map = {'Live': 0}

        phase = 'Train' if train else 'Test'
        phase_path = os.path.join(year_path, sensor, phase)
        if not os.path.isdir(phase_path):
            raise RuntimeError(f"Directory not found: {phase_path}")

        if binary_class:
            self._load_binary(phase_path)
        else:
            self._load_multiclass(phase_path)

    def _load_binary(self, phase_path):
        for label_name, label_id in self.label_map.items():
            data_path = os.path.join(phase_path, label_name)
            if not os.path.isdir(data_path):
                raise RuntimeError(f"Directory not found: {data_path}")

            if label_name == 'Live':
                for img_file in tqdm(os.listdir(data_path), desc=f"Loading Live"):
                    if img_file.endswith(('.png', '.bmp')):
                        image_path = os.path.join(data_path, img_file)
                        self.samples.append((image_path, label_id))
            elif label_name == 'Spoof':
                for material in os.listdir(data_path):
                    material_path = os.path.join(data_path, material)
                    if os.path.isdir(material_path):
                        for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                            if img_file.endswith(('.png', '.bmp')):
                                image_path = os.path.join(material_path, img_file)
                                self.samples.append((image_path, label_id))

    def _load_multiclass(self, phase_path):
        live_path = os.path.join(phase_path, 'Live')
        if not os.path.isdir(live_path):
            raise RuntimeError(f"Directory not found: {live_path}")
        for img_file in tqdm(os.listdir(live_path), desc=f"Loading Live"):
            if img_file.endswith(('.png', '.bmp')):
                image_path = os.path.join(live_path, img_file)
                self.samples.append((image_path, self.label_map['Live']))

        spoof_path = os.path.join(phase_path, 'Spoof')
        if not os.path.isdir(spoof_path):
            raise RuntimeError(f"Directory not found: {spoof_path}")
        materials = sorted([material for material in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, material))])
        
        for i, material in enumerate(materials, 1):
            self.label_map[material] = i
            material_path = os.path.join(spoof_path, material)
            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(material_path, img_file)
                    self.samples.append((image_path, self.label_map[material]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label
    

class CrossSensorDataset(Dataset):
    def __init__(self, year_path, train_sensor, test_sensor, train=True, binary_class=True):
        self.samples = []
        self.binary_class = binary_class
        self.train_sensor = train_sensor
        self.test_sensor = test_sensor
        self.label_map = {}
        
        if binary_class:
            self.label_map = {'Live': 0, 'Spoof': 1}
        else:
            self.label_map = {'Live': 0}

        sensor = train_sensor if train else test_sensor
        sensor_path = os.path.join(year_path, sensor)
        if not os.path.isdir(sensor_path):
            raise RuntimeError(f"Directory not found: {sensor_path}")

        if binary_class:
            self._load_binary(sensor_path)
        else:
            self._load_multiclass(sensor_path)

    def _load_binary(self, sensor_path):
        for phase in ['Train', 'Test']:
            phase_path = os.path.join(sensor_path, phase)
            if not os.path.isdir(phase_path):
                raise RuntimeError(f"Directory not found: {phase_path}")

            for label_name, label_id in self.label_map.items():
                data_path = os.path.join(phase_path, label_name)
                if not os.path.isdir(data_path):
                    raise RuntimeError(f"Directory not found: {data_path}")

                if label_name == 'Live':
                    for img_file in tqdm(os.listdir(data_path), desc=f"Loading Live"):
                        if img_file.endswith(('.png', '.bmp')):
                            image_path = os.path.join(data_path, img_file)
                            self.samples.append((image_path, label_id))
                elif label_name == 'Spoof':
                    for material in os.listdir(data_path):
                        material_path = os.path.join(data_path, material)
                        if os.path.isdir(material_path):
                            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                                if img_file.endswith(('.png', '.bmp')):
                                    image_path = os.path.join(material_path, img_file)
                                    self.samples.append((image_path, label_id))

    def _load_multiclass(self, sensor_path):
        all_materials = set()
        for phase in ['Train', 'Test']:
            spoof_path = os.path.join(sensor_path, phase, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Directory not found: {spoof_path}")
            materials = [material for material in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, material))]
            all_materials.update(materials)
        
        sorted_materials = sorted(list(all_materials))
        for i, material in enumerate(sorted_materials, 1):
            self.label_map[material] = i

        # Load data from both Train and Test
        for phase in ['Train', 'Test']:
            phase_path = os.path.join(sensor_path, phase)
            if not os.path.isdir(phase_path):
                raise RuntimeError(f"Directory not found: {phase_path}")

            live_path = os.path.join(phase_path, 'Live')
            if not os.path.isdir(live_path):
                raise RuntimeError(f"Directory not found: {live_path}")
            for img_file in tqdm(os.listdir(live_path), desc=f"Loading Live"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(live_path, img_file)
                    self.samples.append((image_path, self.label_map['Live']))

            spoof_path = os.path.join(phase_path, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Directory not found: {spoof_path}")
            for material in os.listdir(spoof_path):
                if material in self.label_map:
                    label_id = self.label_map[material]
                    material_path = os.path.join(spoof_path, material)
                    if os.path.isdir(material_path):
                        for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                            if img_file.endswith(('.png', '.bmp')):
                                image_path = os.path.join(material_path, img_file)
                                self.samples.append((image_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label
    

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset: Dataset, val_split: float, seed: int):
    if not 0 < val_split < 1:
        raise ValueError("0 <= val_split <= 1")
    
    generator = torch.Generator().manual_seed(seed)

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    return random_split(dataset, [train_size, val_size], generator=generator)


def create_data_loader(
    year_path: str,
    train_sensor: str,
    test_sensor: str,
    train: bool,
    binary_class: bool,
    transform: dict,
    batch_size: int,
    num_workers: Optional[int] = None,
    val_split: Optional[float] = None,
    seed: Optional[int] = None,
):
    phase = 'Train' if train else 'Test'
    # Intra-sensor
    if train_sensor == test_sensor:
        sensor = train_sensor
        print(f"Creating Intra-sensor {sensor} {phase} dataset")
        dataset = IntraSensorDataset(year_path, sensor, train=train, binary_class=binary_class)
    else: # Cross-sensor
        print(f"Creating Cross-sensor {train_sensor}-{test_sensor} {phase} dataset")
        dataset = CrossSensorDataset(year_path, train_sensor, test_sensor, train=train, binary_class=binary_class)
    label_map = dataset.label_map

    use_pin_memory = True if torch.cuda.is_available() else False
    # Train phase
    if train:
        train_subset, val_subset = split_dataset(dataset, val_split=val_split, seed=seed)
        train_dataset = TransformedDataset(train_subset, transform['Train'])
        val_dataset = TransformedDataset(val_subset, transform['Test'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
        return train_loader, val_loader, label_map
    else: # Test phase
        test_dataset = TransformedDataset(dataset, transform['Test'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
        return test_loader, label_map


def get_classic_data_loaders(config):

    transform = get_transforms(config['TRANSFORM_TYPE'])

    train_loader, val_loader, train_label_map = create_data_loader(
        year_path=config['YEAR_PATH'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        train=True,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT'],
        seed=config['SEED']
    )
    test_loader, test_label_map = create_data_loader(
        year_path=config['YEAR_PATH'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        train=False,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT'],
        seed=config['SEED']
    )
    return train_loader, val_loader, test_loader, train_label_map


def get_multitask_data_loaders(config):

    transform = get_transforms(config['TRANSFORM_TYPE'])

    train_loader, val_loader, train_label_map = create_data_loader(
        year=config['YEAR'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        dataset_path=config['DATASET_PATH'],
        train=True,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT'],
        seed=config['SEED']
    )
    test_loader, test_label_map = create_data_loader(
        year=config['YEAR'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        dataset_path=config['DATASET_PATH'],
        train=False,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT'],
        seed=config['SEED']
    )
    return train_loader, val_loader, test_loader, train_label_map, test_label_map
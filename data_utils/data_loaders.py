import os
from PIL import Image
import sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

class IntraSensorDataset(Dataset):
    def __init__(self, year, sensor, dataset_path, train=True, binary_class=True):
        self.samples = []
        self.label_map = {}
        
        if binary_class:
            self.label_map = {'Live': 0, 'Spoof': 1}
        else:
            self.label_map = {'Live': 0}

        phase = 'Train' if train else 'Test'
        
        print(f"Loading {phase} data for Intra-Sensor ({sensor})...")
        sensor_path = os.path.join(dataset_path, year, sensor, phase)
        if not os.path.isdir(sensor_path):
            raise RuntimeError(f"Dataset directory not found: {sensor_path}")

        if binary_class:
            self._load_binary_data(sensor_path, phase)
        else:
            self._load_multiclass_data(sensor_path, phase)

    def _load_binary_data(self, sensor_path, phase):
        """Load data for binary classification (Live vs Spoof)"""
        for label_name, label_id in self.label_map.items():
            data_path = os.path.join(sensor_path, label_name)
            if not os.path.isdir(data_path):
                raise RuntimeError(f"Data directory not found: {data_path}")

            if label_name == 'Live':
                for img_file in tqdm(os.listdir(data_path), desc=f"Loading {phase} Live"):
                    if img_file.endswith(('.png', '.bmp')):
                        image_path = os.path.join(data_path, img_file)
                        self.samples.append((image_path, label_id))
            elif label_name == 'Spoof':
                for spoof_material in os.listdir(data_path):
                    spoof_material_path = os.path.join(data_path, spoof_material)
                    if os.path.isdir(spoof_material_path):
                        for img_file in tqdm(os.listdir(spoof_material_path), desc=f"Loading {phase} {spoof_material}"):
                            if img_file.endswith(('.png', '.bmp')):
                                image_path = os.path.join(spoof_material_path, img_file)
                                self.samples.append((image_path, label_id))

    def _load_multiclass_data(self, sensor_path, phase):
        """Load data for multiclass classification (Live vs different spoof materials)"""
        # Load Live images
        live_path = os.path.join(sensor_path, 'Live')
        if not os.path.isdir(live_path):
            raise RuntimeError(f"Data directory not found: {live_path}")
        
        for img_file in tqdm(os.listdir(live_path), desc=f"Loading {phase} Live"):
            if img_file.endswith(('.png', '.bmp')):
                image_path = os.path.join(live_path, img_file)
                self.samples.append((image_path, self.label_map['Live']))

        # Load Spoof images and create labels dynamically
        spoof_path = os.path.join(sensor_path, 'Spoof')
        if not os.path.isdir(spoof_path):
            raise RuntimeError(f"Data directory not found: {spoof_path}")

        spoof_materials = sorted([d for d in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, d))])
        
        for i, material in enumerate(spoof_materials, 1):
            self.label_map[material] = i
            material_path = os.path.join(spoof_path, material)
            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {phase} {material}"):
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
    def __init__(self, year, train_sensor, test_sensor, dataset_path, train=True, binary_class=True):
        self.samples = []
        self.binary_class = binary_class
        self.train_sensor = train_sensor
        self.test_sensor = test_sensor
        self.label_map = {}
        
        if binary_class:
            self.label_map = {'Live': 0, 'Spoof': 1}
        else:
            self.label_map = {'Live': 0}

        # Use train_sensor if train is True, otherwise use test_sensor
        sensor = train_sensor if train else test_sensor
        print(f"Loading data for Cross-Sensor (Train: {train_sensor}, Test: {test_sensor})...")

        if binary_class:
            self._load_binary_data(year, sensor, dataset_path)
        else:
            self._load_multiclass_data(year, sensor, dataset_path)

    def _load_binary_data(self, year, sensor, dataset_path):
        """Load data for binary classification (Live vs Spoof)"""
        for phase in ['Train', 'Test']:
            sensor_path = os.path.join(dataset_path, year, sensor, phase)
            if not os.path.isdir(sensor_path):
                raise RuntimeError(f"Dataset directory not found: {sensor_path}")

            for label_name, label_id in self.label_map.items():
                data_path = os.path.join(sensor_path, label_name)
                if not os.path.isdir(data_path):
                    raise RuntimeError(f"Data directory not found: {data_path}")

                if label_name == 'Live':
                    for img_file in tqdm(os.listdir(data_path), desc=f"Loading {sensor} {phase} Live"):
                        if img_file.endswith(('.png', '.bmp')):
                            image_path = os.path.join(data_path, img_file)
                            self.samples.append((image_path, label_id))
                elif label_name == 'Spoof':
                    for spoof_material in os.listdir(data_path):
                        spoof_material_path = os.path.join(data_path, spoof_material)
                        if os.path.isdir(spoof_material_path):
                            for img_file in tqdm(os.listdir(spoof_material_path), desc=f"Loading {sensor} {phase} {spoof_material}"):
                                if img_file.endswith(('.png', '.bmp')):
                                    image_path = os.path.join(spoof_material_path, img_file)
                                    self.samples.append((image_path, label_id))

    def _load_multiclass_data(self, year, sensor, dataset_path):
        """Load data for multiclass classification (Live vs different spoof materials)"""
        # Discover all spoof materials from both Train and Test
        all_spoof_materials = set()
        for phase in ['Train', 'Test']:
            spoof_path = os.path.join(dataset_path, year, sensor, phase, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Spoof directory not found in {phase} set: {spoof_path}")
            materials = [d for d in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, d))]
            all_spoof_materials.update(materials)
        
        sorted_spoof_materials = sorted(list(all_spoof_materials))
        for i, material in enumerate(sorted_spoof_materials, 1):
            self.label_map[material] = i

        # Load data from both Train and Test
        for phase in ['Train', 'Test']:
            sensor_path = os.path.join(dataset_path, year, sensor, phase)
            if not os.path.isdir(sensor_path):
                raise RuntimeError(f"Dataset directory not found: {sensor_path}")

            # Load Live images
            live_path = os.path.join(sensor_path, 'Live')
            if not os.path.isdir(live_path):
                raise RuntimeError(f"Live directory not found in {phase} set: {live_path}")
            for img_file in tqdm(os.listdir(live_path), desc=f"Loading {sensor} {phase} Live"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(live_path, img_file)
                    self.samples.append((image_path, self.label_map['Live']))

            # Load Spoof images
            spoof_path = os.path.join(sensor_path, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Spoof directory not found in {phase} set: {spoof_path}")
            for material in os.listdir(spoof_path):
                if material in self.label_map:
                    label_id = self.label_map[material]
                    material_path = os.path.join(spoof_path, material)
                    if os.path.isdir(material_path):
                        for img_file in tqdm(os.listdir(material_path), desc=f"Loading {sensor} {phase} {material}"):
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


def split_dataset(dataset: Dataset, val_split: float = 0.2, seed: int = 42):
    if not 0 < val_split < 1:
        raise ValueError("Validation split must be between 0 and 1.")

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def get_dataloader(
    year: str,
    train_sensor: str,
    test_sensor: str,
    dataset_path: str,
    train: bool,
    binary_class: bool,
    transform: dict,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int = 42,
):
    # Use intra-dataset if train_sensor equals test_sensor, cross-dataset otherwise
    if train_sensor == test_sensor:
        # For intra-sensor, use the same sensor regardless of train/test phase
        sensor = train_sensor  # Since train_sensor == test_sensor
        dataset = IntraSensorDataset(year, sensor, dataset_path, train=train, binary_class=binary_class)
    else: # Cross-sensor
        dataset = CrossSensorDataset(year, train_sensor, test_sensor, dataset_path, train=train, binary_class=binary_class)
    label_map = dataset.label_map

    if train:
        train_subset, val_subset = split_dataset(dataset, val_split=val_split, seed=seed)
        train_dataset = TransformedDataset(train_subset, transform['Train'])
        val_dataset = TransformedDataset(val_subset, transform['Test'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, label_map
    else: # Test phase
        test_dataset = TransformedDataset(dataset, transform['Test'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader, label_map
import torchvision.transforms as T

basic_transform = {
    'Train': T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=(-20, 20),          # Rotation
            translate=(0.2, 0.2),       # Horizontal/vertical shift
            shear=(-20, 20),            # Shear
            scale=(0.8, 1.2),           # Zoom
            interpolation=T.InterpolationMode.NEAREST,
            fill=0
        ),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),

    'Test': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

siglip2_transform = {
    'Train': T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=(-20, 20),          # Rotation
            translate=(0.2, 0.2),       # Horizontal/vertical shift
            shear=(-20, 20),            # Shear
            scale=(0.8, 1.2),           # Zoom
            interpolation=T.InterpolationMode.BILINEAR,  # Use BILINEAR for ViT
            fill=0
        ),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # SigLIP 2 normalization
    ]),

    'Test': T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # SigLIP 2 normalization
    ])
}

def get_transforms(transform_type):
    if transform_type == 'basic':
        return basic_transform
    elif transform_type == 'siglip2':
        return siglip2_transform
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}. Supported types: 'basic', 'siglip2'")


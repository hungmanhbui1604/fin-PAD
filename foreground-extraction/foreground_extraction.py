import numpy as np
import cv2 as cv
import os
from PIL import Image
from tqdm import tqdm
import argparse


def extract_one(
    image_path: str, 
    block_size: int, 
    delta: int, 
    kernel_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer greater than 1.")
    
    # Load image
    pil = Image.open(image_path).convert('L') # Convert to grayscale
    original = np.array(pil).astype(np.uint8)

    # Binarize the image using adaptive thresholding to create a black and white mask.
    binarized = cv.adaptiveThreshold(
        src=original,
        maxValue=1,
        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=block_size,
        C=delta
    )

    # Create a kernel for morphological operations.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Dilate the binarized image to connect broken ridges and fill small holes.
    dilated = cv.dilate(binarized, kernel, iterations=1)

    # Find all connected components (i.e., separate white regions) in the dilated image.
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(dilated, 8, cv.CV_32S)

    # If there's only the background component, return early.
    if num_labels <= 1:
        return original, binarized, dilated, dilated, original
    
    # Find the label of the largest component (ignoring the background at index 0).
    areas = stats[1:, cv.CC_STAT_AREA]
    foreground_label = np.argmax(areas) + 1

    # Create a mask containing only the largest component.
    foreground_mask = np.zeros_like(dilated, dtype=np.uint8)
    foreground_mask[labels == foreground_label] = 1

    # Get the bounding box coordinates of the largest component.
    x = stats[foreground_label, cv.CC_STAT_LEFT]
    y = stats[foreground_label, cv.CC_STAT_TOP]
    w = stats[foreground_label, cv.CC_STAT_WIDTH]
    h = stats[foreground_label, cv.CC_STAT_HEIGHT]
    # Crop the original image to the bounding box of the foreground.
    foreground = original[y:y+h, x:x+w]

    return original, binarized, dilated, foreground_mask, foreground


def extract_all(
    data_dir: str,
    out_dir: str,
    block_size: int,
    delta: int,
    kernel_size: int
) -> None:
    # Supported image extensions
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    for dirpath, _, filenames in os.walk(data_dir):
        # Create corresponding directory in the out dir
        relative_path = os.path.relpath(dirpath, data_dir)
        if relative_path == '.':
            relative_path = ''
        current_output_dir = os.path.join(out_dir, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)

        image_files = []
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(dirpath, filename))

        if not image_files:
            continue

        display_path = os.path.basename(data_dir)
        if relative_path != '.':
            display_path = os.path.join(display_path, relative_path)

        print(f"Found {len(image_files)} images in '{display_path}'")
        
        for image_path in tqdm(image_files, desc=f"Processing images in '{display_path}'"):
            try:
                # Extract foreground using the extract_one() function
                _, _, _, _, foreground = extract_one(image_path, block_size, delta, kernel_size)
                
                # Save the foreground image to the out dir
                output_path = os.path.join(current_output_dir, os.path.basename(image_path))
                Image.fromarray(foreground).save(output_path)
                
            except Exception as e:
                # Use tqdm.write to print messages without breaking the progress bar
                tqdm.write(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                
                # Skip the error image and continue with the next one
                continue
    
    print(f"Processing complete! Foreground images saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fingerprint Foreground Extraction')
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='Input directory containing fingerprint images')
    parser.add_argument('-o', '--out_dir', type=str, default='./foregrounds',
                        help='Output directory for foreground images (default: ./foregrounds)')
    parser.add_argument('-b', '--block_size', type=int, default=3,
                        help='Block size for adaptive thresholding (must be an odd integer > 1, default: 3)')
    parser.add_argument('-d', '--delta', type=int, default=2,
                        help='Delta value for adaptive thresholding (default: 2)')
    parser.add_argument('-k', '--kernel_size', type=int, default=3,
                        help='Kernel size for morphological operations (default: 3)')
    
    args = parser.parse_args()

    extract_all(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        block_size=args.block_size,
        delta=args.delta,
        kernel_size=args.kernel_size
    )

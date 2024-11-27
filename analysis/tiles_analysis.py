import os
import json
import cv2
import numpy as np
import argparse
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tile Analysis Script")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/izar/dlopez/ml4science/data/",
        help="Base directory for data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["BACH", "BRACS"],
        default="BACH",
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Name of the image file to process (e.g., 'b001.tif'). If not specified, process a number of images specified by --num_images.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of images to process for debugging. Ignored if --image_name is specified.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/output/",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["INFO", "DEBUG"],
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()

def configure_logging(log_level: str):
    """Configure logging level."""
    level = logging.DEBUG if log_level == "DEBUG" else logging.INFO
    logger.setLevel(level)

def get_augmentation_levels(dataset: str) -> List[str]:
    """Get augmentation levels based on the dataset."""
    if dataset == "BACH":
        return ["5x", "10x", "20x"]
    elif dataset == "BRACS":
        return ["5x", "10x", "20x", "40x"]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

def draw_tiles_on_image(
    image: np.ndarray,
    tiles: List[Tuple[int, int]],
    tile_size: int,
    color=(0, 255, 0),
    thickness=2,
) -> np.ndarray:
    """
    Draw rectangles on the image based on tile coordinates and size.

    Args:
        image: The original image.
        tiles: List of tile coordinates (top-left corners).
        tile_size: Size of each tile.
        color: Rectangle color.
        thickness: Rectangle border thickness.

    Returns:
        Image with rectangles drawn.
    """
    for x, y in tiles:
        top_left = (int(x), int(y))
        bottom_right = (int(x + tile_size), int(y + tile_size))
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def rectangles_overlap(
    rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]
) -> bool:
    """
    Check if two rectangles overlap.

    Args:
        rect1: Coordinates of the first rectangle (x_min, y_min, x_max, y_max).
        rect2: Coordinates of the second rectangle.

    Returns:
        True if rectangles overlap, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return overlap_x > 0 and overlap_y > 0

def check_overlaps(
    tiles: List[Tuple[int, int]], tile_size: int
) -> List[Tuple[int, int]]:
    """
    Check for overlapping tiles.

    Args:
        tiles: List of tile coordinates.
        tile_size: Size of each tile.

    Returns:
        List of tuples indicating indices of overlapping tiles.
    """
    overlaps = []
    rectangles = []
    for idx, (x, y) in enumerate(tiles):
        rect = (x, y, x + tile_size, y + tile_size)
        rectangles.append((idx, rect))
    for i in range(len(rectangles)):
        idx1, rect1 = rectangles[i]
        for j in range(i + 1, len(rectangles)):
            idx2, rect2 = rectangles[j]
            if rectangles_overlap(rect1, rect2):
                overlaps.append((idx1, idx2))
    return overlaps

def process_image(
    image_name: str,
    images_dir: str,
    dataset_dir: str,
    augmentation_levels: List[str],
    output_dir: str,
):
    """
    Process a single image: draw tiles and check for overlaps.

    Args:
        image_name: Name of the image file.
        images_dir: Directory containing images.
        dataset_dir: Base dataset directory.
        augmentation_levels: List of augmentation levels to process.
        output_dir: Directory to save output images.
    """
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not read image {image_name}")
        return

    for augmentation_level in augmentation_levels:
        tiles_metadata_dir = os.path.join(
            dataset_dir, f"tiles_metadata_224_{augmentation_level}"
        )
        metadata_file = os.path.join(tiles_metadata_dir, f"{image_name}.json")

        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata file {metadata_file} does not exist")
            continue

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        tiles = metadata["tiles"]
        patch_size = metadata["patch_size"]
        mpp = metadata["mpp"]
        base_mpp = metadata["base_mpp"]
        scaling_factor = mpp / base_mpp
        tile_size = round(patch_size * scaling_factor)

        # Draw tiles on image
        image_with_tiles = image.copy()
        image_with_tiles = draw_tiles_on_image(image_with_tiles, tiles, tile_size)

        # Check for overlaps
        overlaps = check_overlaps(tiles, tile_size)

        # Log coordinates and overlaps
        logger.info(f"Image: {image_name}, Augmentation Level: {augmentation_level}")
        logger.info("Tile coordinates:")
        for idx, (x, y) in enumerate(tiles):
            logger.debug(f"Tile {idx}: ({x}, {y})")
        if overlaps:
            logger.info("Overlaps detected between tiles:")
            for i, j in overlaps:
                logger.info(f"Tiles {i} and {j} overlap")
        else:
            logger.info("No overlaps detected between tiles")

        # Save image
        output_image_name = (
            f"{os.path.splitext(image_name)[0]}_{augmentation_level}_tiles.png"
        )
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, image_with_tiles)
        logger.info(f"Saved image with tiles to {output_image_path}\n")

def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_name:
        images_list = [args.image_name]
    else:
        # Get a list of images for processing
        images_list = [
            f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))
        ][: args.num_images]

    try:
        augmentation_levels = get_augmentation_levels(args.dataset)
    except ValueError as e:
        logger.error(e)
        return

    for image_name in images_list:
        process_image(
            image_name, images_dir, dataset_dir, augmentation_levels, args.output_dir
        )

if __name__ == "__main__":
    main()
